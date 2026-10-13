"""
Krushi Mithra
Smart Farming & Intelligent Irrigation System

control.py
==========

Central irrigation control and pump safety layer.

Responsibilities:
- Evaluate AUTO irrigation decisions.
- Respect SYSTEM ON/OFF state.
- Respect AUTO/MANUAL mode.
- Apply rain protection.
- Use M2 irrigation prediction.
- Use M3 future soil-moisture prediction.
- Use M4 future rainfall prediction.
- Apply critical-soil safety rule.
- Send final pump command to ESP32.
- Never directly control GPIO.
- Never contain Wi-Fi credentials.
- Never load ML models.

ESP32 remains responsible for:
- Relay GPIO23
- Physical pump
- Hardware safety
- Final enforcement of system/mode rules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from esp32 import ESP32, ESP32Error
from config import (
    SOIL_TRIGGER,
    CRITICAL_SOIL,
    RAIN_PROTECTION,
)


logger = logging.getLogger("KrushiMithra.Control")


# ============================================================
# CONTROL EXCEPTIONS
# ============================================================

class ControlError(Exception):
    """Base exception for irrigation control errors."""


class ControlCommunicationError(ControlError):
    """Raised when ESP32 communication fails."""


# ============================================================
# CONTROL RESULT
# ============================================================

@dataclass
class ControlDecision:
    """
    Final irrigation decision before/after sending the command.
    """

    pump_required: bool
    pump_command: bool
    reason: str

    system_enabled: bool
    mode: str

    soil_moisture_percent: Optional[float] = None
    future_soil_moisture_percent: Optional[float] = None
    predicted_rainfall_mm: Optional[float] = None

    m2_irrigation: Optional[str] = None
    rain_detected: bool = False
    rain_wetness_raw: Optional[float] = None

    command_sent: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pump_required": self.pump_required,
            "pump_command": self.pump_command,
            "reason": self.reason,
            "system_enabled": self.system_enabled,
            "mode": self.mode,
            "soil_moisture_percent": self.soil_moisture_percent,
            "future_soil_moisture_percent": (
                self.future_soil_moisture_percent
            ),
            "predicted_rainfall_mm": self.predicted_rainfall_mm,
            "m2_irrigation": self.m2_irrigation,
            "rain_detected": self.rain_detected,
            "rain_wetness_raw": self.rain_wetness_raw,
            "command_sent": self.command_sent,
            "error": self.error,
        }


# ============================================================
# IRRIGATION CONTROLLER
# ============================================================

class IrrigationController:
    """
    Central irrigation controller.

    ML models make predictions.

    This class decides what to do with those predictions.

    It does NOT:
        - load models
        - communicate with weather APIs
        - read sensors directly
        - control GPIO
        - manage Wi-Fi
    """

    def __init__(
        self,
        esp32: ESP32,
        soil_trigger: float = SOIL_TRIGGER,
        critical_soil: float = CRITICAL_SOIL,
        rain_protection: bool = RAIN_PROTECTION,
    ):
        self.esp32 = esp32

        self.soil_trigger = float(soil_trigger)
        self.critical_soil = float(critical_soil)
        self.rain_protection = bool(rain_protection)

        self._validate_config()

        self.last_decision: Optional[ControlDecision] = None
        self._pump_retry_state: Optional[bool] = None
        self._pump_retry_count = 0
        self._pump_retry_after = 0.0
        self._last_successful_pump_state: Optional[bool] = None
        self._last_pump_success_at = 0.0

        logger.info(
            "Irrigation controller initialized: "
            "soil_trigger=%.2f, critical_soil=%.2f, "
            "rain_protection=%s",
            self.soil_trigger,
            self.critical_soil,
            self.rain_protection,
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def _validate_config(self) -> None:
        if not 0 <= self.soil_trigger <= 100:
            raise ValueError(
                "soil_trigger must be between 0 and 100."
            )

        if not 0 <= self.critical_soil <= 100:
            raise ValueError(
                "critical_soil must be between 0 and 100."
            )

        if self.critical_soil > self.soil_trigger:
            raise ValueError(
                "critical_soil cannot be greater than soil_trigger."
            )

    def update_config(
        self,
        soil_trigger: Optional[float] = None,
        critical_soil: Optional[float] = None,
        rain_protection: Optional[bool] = None,
    ) -> None:
        """
        Update local control configuration.
        """

        if soil_trigger is not None:
            self.soil_trigger = float(soil_trigger)

        if critical_soil is not None:
            self.critical_soil = float(critical_soil)

        if rain_protection is not None:
            self.rain_protection = bool(rain_protection)

        self._validate_config()

        logger.info(
            "Control configuration updated: "
            "soil_trigger=%.2f, critical_soil=%.2f, "
            "rain_protection=%s",
            self.soil_trigger,
            self.critical_soil,
            self.rain_protection,
        )

    def sync_config_from_esp32(self) -> Dict[str, Any]:
        """
        Read the current irrigation configuration from ESP32
        and synchronize the local controller.
        """

        config = self.esp32.get_config()

        if "soil_trigger" in config:
            self.soil_trigger = float(config["soil_trigger"])

        if "critical_soil" in config:
            self.critical_soil = float(config["critical_soil"])

        if "rain_protection" in config:
            self.rain_protection = bool(config["rain_protection"])

        self._validate_config()

        return config

    # ========================================================
    # VALUE EXTRACTION
    # ========================================================

    @staticmethod
    def _get_soil_moisture(
        esp32_data: Dict[str, Any],
    ) -> Optional[float]:
        """
        Get soil moisture percentage.

        ESP32 normally provides soil_moisture_percent.

        Raw conversion is handled only as a fallback.
        """

        value = esp32_data.get("soil_moisture_percent")

        if value is not None:
            return float(value)

        raw = esp32_data.get("soil_moisture_raw")

        if raw is None:
            return None

        raw = float(raw)

        dry = 4095.0
        wet = 1500.0

        if raw >= dry:
            return 0.0

        if raw <= wet:
            return 100.0

        percentage = (
            (dry - raw) / (dry - wet)
        ) * 100.0

        return max(0.0, min(100.0, percentage))

    @staticmethod
    def _get_future_soil_moisture(
        m3_result: Optional[Dict[str, Any]],
    ) -> Optional[float]:

        if not m3_result:
            return None

        value = m3_result.get(
            "future_soil_moisture_percent"
        )

        if value is None:
            return None

        return float(value)

    @staticmethod
    def _get_m2_irrigation(
        m2_result: Optional[Dict[str, Any]],
    ) -> Optional[str]:

        if not m2_result:
            return None

        value = m2_result.get("irrigation")

        if value is None:
            return None

        return str(value).strip()

    @staticmethod
    def _get_rain_wetness_raw(esp32_data: Dict[str, Any]) -> Optional[float]:
        value = esp32_data.get("rain_wetness_raw")
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _is_rain_detected(
        esp32_data: Dict[str, Any],
    ) -> bool:
        """
        FC-37 is treated as a rain/wetness detection signal.

        It is NOT interpreted as rainfall in millimetres.
        """

        value = esp32_data.get("rain_detected")

        if value is None:
            value = 0

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in (
                "1",
                "true",
                "yes",
                "rain",
                "detected",
            )

        return bool(value)

    # ========================================================
    # PREDICTION INTERPRETATION
    # ========================================================

    @staticmethod
    def _irrigation_requested(
        m2_irrigation: Optional[str],
    ) -> bool:
        """
        Convert the M2 prediction into a boolean decision.
        """

        if m2_irrigation is None:
            return False

        value = m2_irrigation.strip().lower()

        irrigation_values = {
            "yes",
            "true",
            "1",
            "irrigate",
            "irrigation",
            "required",
            "required irrigation",
            "need irrigation",
            "needs irrigation",
            "on",
        }

        return value in irrigation_values

    # ========================================================
    # DECISION ENGINE
    # ========================================================

    def evaluate(
        self,
        esp32_data: Dict[str, Any],
        m2_result: Optional[Dict[str, Any]] = None,
        m3_result: Optional[Dict[str, Any]] = None,
        m4_result: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None,
    ) -> ControlDecision:
        """
        Evaluate whether the pump should be ON or OFF.

        Only physical rain detection and soil conditions are used to drive
        irrigation decisions. M4 rainfall forecasts are informational only and
        must not be interpreted as a pump safety input.
        """

        _ = m4_result
        status = status or {}

        system_enabled = bool(
            esp32_data.get(
                "system_enabled",
                status.get(
                    "system_enabled",
                    False,
                ),
            )
        )

        mode = str(
            esp32_data.get(
                "mode",
                status.get(
                    "mode",
                    "AUTO",
                ),
            )
        ).upper()

        pump_current = bool(
            esp32_data.get(
                "pump",
                status.get(
                    "pump",
                    False,
                ),
            )
        )

        soil_moisture = self._get_soil_moisture(
            esp32_data
        )

        future_soil_moisture = (
            self._get_future_soil_moisture(
                m3_result
            )
        )

        m2_irrigation = self._get_m2_irrigation(
            m2_result
        )

        rain_detected = self._is_rain_detected(
            esp32_data
        )
        rain_wetness_raw = self._get_rain_wetness_raw(esp32_data)

        # ----------------------------------------------------
        # 1. SYSTEM OFF
        # ----------------------------------------------------

        if not system_enabled:
            return ControlDecision(
                pump_required=False,
                pump_command=False,
                reason="SYSTEM_OFF",
                system_enabled=False,
                mode=mode,
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        # ----------------------------------------------------
        # 2. MANUAL MODE
        # ----------------------------------------------------

        if mode == "MANUAL":
            return ControlDecision(
                pump_required=pump_current,
                pump_command=pump_current,
                reason="MANUAL_MODE",
                system_enabled=True,
                mode="MANUAL",
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        # ----------------------------------------------------
        # 3. UNKNOWN MODE
        # ----------------------------------------------------

        if mode != "AUTO":
            return ControlDecision(
                pump_required=False,
                pump_command=False,
                reason="INVALID_MODE_SAFE_OFF",
                system_enabled=True,
                mode=mode,
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        # ----------------------------------------------------
        # 4. RAIN PROTECTION
        # ----------------------------------------------------

        if self.rain_protection and rain_detected:
            return ControlDecision(
                pump_required=False,
                pump_command=False,
                reason="RAIN_DETECTED",
                system_enabled=True,
                mode="AUTO",
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=True,
            )

        # ----------------------------------------------------
        # 5. MISSING SOIL DATA
        # ----------------------------------------------------

        if soil_moisture is None:
            return ControlDecision(
                pump_required=False,
                pump_command=False,
                reason="SOIL_MOISTURE_UNAVAILABLE_SAFE_OFF",
                system_enabled=True,
                mode="AUTO",
                soil_moisture_percent=None,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        # ----------------------------------------------------
        # 6. CRITICAL SOIL MOISTURE
        # ----------------------------------------------------

        if soil_moisture <= self.critical_soil:
            return ControlDecision(
                pump_required=True,
                pump_command=True,
                reason="CRITICAL_SOIL_MOISTURE",
                system_enabled=True,
                mode="AUTO",
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        # Keep an active pump running through a small recovery band so
        # sensor noise around soil_trigger does not chatter the relay.
        if pump_current and soil_moisture < min(100.0, self.soil_trigger + 3.0):
            return ControlDecision(
                pump_required=True,
                pump_command=True,
                reason="PUMP_HYSTERESIS_HOLD",
                system_enabled=True,
                mode="AUTO",
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        # ----------------------------------------------------
        # 7. M2 IRRIGATION PREDICTION
        # ----------------------------------------------------

        irrigation_requested = (
            self._irrigation_requested(
                m2_irrigation
            )
        )

        future_soil_low = False

        if future_soil_moisture is not None:
            future_soil_low = (
                future_soil_moisture
                <= self.soil_trigger
            )

        if (
            irrigation_requested
            and soil_moisture <= self.soil_trigger
        ):
            return ControlDecision(
                pump_required=True,
                pump_command=True,
                reason="M2_IRRIGATION_AND_LOW_SOIL",
                system_enabled=True,
                mode="AUTO",
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        if (
            future_soil_low
            and soil_moisture <= self.soil_trigger
        ):
            return ControlDecision(
                pump_required=True,
                pump_command=True,
                reason="FUTURE_SOIL_MOISTURE_LOW",
                system_enabled=True,
                mode="AUTO",
                soil_moisture_percent=soil_moisture,
                future_soil_moisture_percent=future_soil_moisture,
                predicted_rainfall_mm=None,
                m2_irrigation=m2_irrigation,
                rain_detected=rain_detected,
                rain_wetness_raw=rain_wetness_raw,
            )

        return ControlDecision(
            pump_required=False,
            pump_command=False,
            reason="SOIL_MOISTURE_ADEQUATE",
            system_enabled=True,
            mode="AUTO",
            soil_moisture_percent=soil_moisture,
            future_soil_moisture_percent=future_soil_moisture,
            predicted_rainfall_mm=None,
            m2_irrigation=m2_irrigation,
            rain_detected=rain_detected,
        )

    # ========================================================
    # APPLY DECISION TO ESP32
    # ========================================================

    def apply_decision(
        self,
        decision: ControlDecision,
        current_pump_state: Optional[bool] = None,
    ) -> ControlDecision:
        """
        Send the final AUTO pump command to ESP32.

        AUTO commands are explicitly marked as coming from PC.
        """

        now = time.monotonic()

        if (
            current_pump_state is not None
            and bool(current_pump_state) == bool(decision.pump_command)
        ):
            self._pump_retry_state = None
            self._pump_retry_count = 0
            self._pump_retry_after = 0.0
            decision.command_sent = False
            logger.debug(
                "Pump command skipped: desired state %s == current state %s | reason=%s",
                "ON" if decision.pump_command else "OFF",
                "ON" if current_pump_state else "OFF",
                decision.reason,
            )
            self.last_decision = decision
            return decision

        if self._last_successful_pump_state == decision.pump_command:
            elapsed = now - self._last_pump_success_at
            if current_pump_state is None or bool(current_pump_state) == decision.pump_command:
                decision.command_sent = False
                self.last_decision = decision
                return decision
            if elapsed < 30.0:
                decision.command_sent = False
                logger.debug(
                    "Pump resynchronization deferred for %.1fs after successful command.",
                    30.0 - elapsed,
                )
                self.last_decision = decision
                return decision

        if (
            self._pump_retry_state == decision.pump_command
            and now < self._pump_retry_after
        ):
            decision.command_sent = False
            logger.debug(
                "Pump command retry deferred for %.1fs after communication failure.",
                self._pump_retry_after - now,
            )
            self.last_decision = decision
            return decision

        try:
            self.esp32.set_pump(
                decision.pump_command,
                source="pc",
            )

            decision.command_sent = True
            self._last_successful_pump_state = decision.pump_command
            self._last_pump_success_at = time.monotonic()
            self._pump_retry_state = None
            self._pump_retry_count = 0
            self._pump_retry_after = 0.0

            logger.info(
                "Pump command sent: %s | source=pc | reason=%s",
                "ON" if decision.pump_command else "OFF",
                decision.reason,
            )

            self.last_decision = decision

            return decision

        except ESP32Error as exc:
            if self._pump_retry_state == decision.pump_command:
                self._pump_retry_count += 1
            else:
                self._pump_retry_state = decision.pump_command
                self._pump_retry_count = 1
            retry_delay = min(
                2 ** (self._pump_retry_count - 1),
                30,
            )
            self._pump_retry_after = time.monotonic() + retry_delay
            decision.command_sent = False
            decision.error = str(exc)

            logger.error(
                "Failed to send pump command: %s",
                exc,
            )

            raise ControlCommunicationError(
                f"Failed to apply pump decision: {exc}"
            ) from exc

    # ========================================================
    # ONE COMPLETE AUTO CONTROL CYCLE
    # ========================================================

    def run_auto_cycle(
        self,
        esp32_data: Optional[Dict[str, Any]] = None,
        status: Optional[Dict[str, Any]] = None,
        m2_result: Optional[Dict[str, Any]] = None,
        m3_result: Optional[Dict[str, Any]] = None,
        m4_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one complete irrigation control cycle.

        If ESP32 data/status are not supplied, they are read
        directly from ESP32.
        """

        try:
            if esp32_data is None:
                esp32_data = self.esp32.get_data()

            if status is None:
                status = self.esp32.get_status()

            decision = self.evaluate(
                esp32_data=esp32_data,
                m2_result=m2_result,
                m3_result=m3_result,
                m4_result=m4_result,
                status=status,
            )

            current_pump_state = esp32_data.get(
                "pump",
                status.get("pump"),
            )
            if current_pump_state is not None:
                current_pump_state = bool(current_pump_state)

            decision = self.apply_decision(
                decision,
                current_pump_state=current_pump_state,
            )

            return {
                "success": True,
                "decision": decision.to_dict(),
                "sensor_data": esp32_data,
                "status": status,
            }

        except ControlError:
            raise

        except ESP32Error as exc:
            logger.error(
                "ESP32 communication failed during "
                "control cycle: %s",
                exc,
            )

            raise ControlCommunicationError(
                f"ESP32 communication failed: {exc}"
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected control cycle failure."
            )

            raise ControlError(
                f"Control cycle failed: {exc}"
            ) from exc

    # ========================================================
    # MANUAL CONTROL
    # ========================================================

    def manual_pump(
        self,
        state: bool,
    ) -> Dict[str, Any]:
        """
        Manual pump command.

        Explicitly marked as source='manual'.
        The ESP32 still enforces MANUAL mode.
        """

        try:
            result = self.esp32.set_pump(
                bool(state),
                source="manual",
            )

            logger.info(
                "Manual pump command: %s | source=manual",
                "ON" if state else "OFF",
            )

            return {
                "success": True,
                "pump": bool(state),
                "source": "manual",
                "esp32_response": result,
            }

        except ESP32Error as exc:
            logger.error(
                "Manual pump command failed: %s",
                exc,
            )

            raise ControlCommunicationError(
                f"Manual pump command failed: {exc}"
            ) from exc

    # ========================================================
    # SYSTEM CONTROL
    # ========================================================

    def set_system(
        self,
        enabled: bool,
    ) -> Dict[str, Any]:
        """
        Enable/disable Krushi Mithra automation.
        """

        try:
            result = self.esp32.set_system(
                bool(enabled)
            )

            logger.info(
                "System %s.",
                "ENABLED" if enabled else "DISABLED",
            )

            return result

        except ESP32Error as exc:
            raise ControlCommunicationError(
                f"Failed to change system state: {exc}"
            ) from exc

    # ========================================================
    # MODE CONTROL
    # ========================================================

    def set_mode(
        self,
        mode: str,
    ) -> Dict[str, Any]:
        """
        Change AUTO/MANUAL mode.
        """

        mode = str(mode).strip().upper()

        if mode not in ("AUTO", "MANUAL"):
            raise ValueError(
                "Mode must be AUTO or MANUAL."
            )

        try:
            result = self.esp32.set_mode(mode)

            logger.info(
                "System mode changed to %s.",
                mode,
            )

            return result

        except ESP32Error as exc:
            raise ControlCommunicationError(
                f"Failed to change mode: {exc}"
            ) from exc

    # ========================================================
    # EMERGENCY / SAFE OFF
    # ========================================================

    def safe_pump_off(
        self,
        reason: str = "SAFE_OFF",
    ) -> Dict[str, Any]:
        """
        Force a pump OFF command.

        Useful during:
        - startup
        - shutdown
        - communication recovery
        - unexpected errors
        """

        try:
            result = self.esp32.set_pump(
                False,
                source="pc",
            )

            logger.warning(
                "Pump forced OFF: %s",
                reason,
            )

            return {
                "success": True,
                "pump": False,
                "source": "pc",
                "reason": reason,
                "esp32_response": result,
            }

        except ESP32Error as exc:
            logger.error(
                "Failed to force pump OFF: %s",
                exc,
            )

            raise ControlCommunicationError(
                f"Failed to force pump OFF: {exc}"
            ) from exc

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Return current ESP32 status.
        """

        try:
            return self.esp32.get_status()

        except ESP32Error as exc:
            raise ControlCommunicationError(
                f"Failed to read ESP32 status: {exc}"
            ) from exc

    def get_last_decision(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the most recent control decision.
        """

        if self.last_decision is None:
            return None

        return self.last_decision.to_dict()


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    print("=" * 60)
    print("KRUSHI MITHRA - CONTROL TEST")
    print("=" * 60)

    try:
        esp32 = ESP32()

        controller = IrrigationController(
            esp32=esp32
        )

        print("\n[ESP32 STATUS]")

        status = controller.get_status()
        print(status)

        print("\n[SENSOR DATA]")

        sensor_data = esp32.get_data()
        print(sensor_data)

        print("\n[CONTROL CONFIG]")

        config = controller.sync_config_from_esp32()
        print(config)

        print("\n[TEST DECISION]")

        # No ML prediction is generated here.
        # This only tests the control/safety decision layer.

        decision = controller.evaluate(
            esp32_data=sensor_data,
            status=status,
            m2_result=None,
            m3_result=None,
            m4_result=None,
        )

        print(decision.to_dict())

        print("\n[NOTE]")
        print(
            "This test does NOT automatically send "
            "the pump command."
        )

        print("\nControl test completed.")

    except Exception as exc:

        print("\n[ERROR]")
        print(str(exc))