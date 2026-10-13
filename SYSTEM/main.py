"""
Krushi Mithra
Smart Farming & Intelligent Irrigation System

main.py
=======

Central runtime orchestrator.

Responsibilities:
- Connect to ESP32.
- Load ML engine once.
- Initialize weather service.
- Continuously read ESP32 sensor data.
- Run M2 irrigation prediction.
- Run M3 future soil-moisture prediction.
- Run M4 future rainfall prediction.
- Run irrigation control.
- Keep the runtime alive.
- Handle communication/runtime failures safely.

main.py does NOT:
- Control GPIO directly.
- Manage Wi-Fi credentials.
- Load models repeatedly.
- Serve Flask routes.
- Implement irrigation decision rules.
"""

from __future__ import annotations
from backend import (
    app as backend_app,
    update_cached_esp32_data,
)

import logging
import signal
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import (
    SYSTEM_NAME,
    SYSTEM_VERSION,
    TIMEZONE,
    ESP32_IP,
    ESP32_POLL_INTERVAL,
    ESP32_STATUS_POLL_INTERVAL,
    BACKEND_HEARTBEAT_INTERVAL,
    M2_PREDICTION_INTERVAL,
    M3_PREDICTION_INTERVAL,
    M4_PREDICTION_INTERVAL,
    WEATHER_CURRENT_INTERVAL,
    M4_SYNC_HOUR_1,
    M4_SYNC_HOUR_2,
    TRAINING_END_DATE,
    RUNTIME_START_DATE,
    M1_LOCATION,
    M1_SOIL_TYPE,
    M1_IRRIGATION,
)

from esp32 import (
    ESP32,
    ESP32Error,
)

from ml import (
    get_ml_engine,
)

from weather import (
    M4WeatherHistory,
    ensure_m4_history_ready,
    get_current_weather,
    schedule_m4_sync,
    sync_m1_weather,
)

from control import (
    IrrigationController,
    ControlError,
    ControlCommunicationError,
)


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("KrushiMithra.Main")


# ============================================================
# RUNTIME STATE
# ============================================================

class RuntimeState:
    """
    Shared runtime state.

    Threads update this object.
    A lock protects shared data.
    """

    def __init__(self):
        self.lock = threading.RLock()

        # ----------------------------------------------------
        # ESP32
        # ----------------------------------------------------

        self.esp32_online = False
        self.esp32_data: Dict[str, Any] = {}
        self.esp32_status: Dict[str, Any] = {}
        self.esp32_config: Dict[str, Any] = {}

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        self.weather: Dict[str, Any] = {}

        # ----------------------------------------------------
        # ML RESULTS
        # ----------------------------------------------------

        self.m1_result: Optional[Dict[str, Any]] = None
        self.m2_result: Optional[Dict[str, Any]] = None
        self.m3_result: Optional[Dict[str, Any]] = None
        self.m4_result: Optional[Dict[str, Any]] = None

        # ----------------------------------------------------
        # CONTROL
        # ----------------------------------------------------

        self.control_result: Optional[Dict[str, Any]] = None

        # ----------------------------------------------------
        # USER / CROP SELECTION
        # ----------------------------------------------------

        self.selected_crop: Optional[str] = None
        self.best_time: str = "Any"

        self.location: Optional[str] = M1_LOCATION
        self.soil_type: Optional[str] = M1_SOIL_TYPE
        self.irrigation: Optional[str] = M1_IRRIGATION

        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        self.running = True

    # ========================================================
    # THREAD-SAFE SNAPSHOTS
    # ========================================================

    def get_esp32_data(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.esp32_data)

    def get_esp32_status(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.esp32_status)

    def get_weather(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.weather)

    def get_m2(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return (
                dict(self.m2_result)
                if self.m2_result
                else None
            )

    def get_m3(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return (
                dict(self.m3_result)
                if self.m3_result
                else None
            )

    def get_m4(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return (
                dict(self.m4_result)
                if self.m4_result
                else None
            )


# ============================================================
# MAIN RUNTIME
# ============================================================

class KrushiMithraRuntime:

    def __init__(self):

        self.state = RuntimeState()

        self.stop_event = threading.Event()

        self.threads: list[threading.Thread] = []

        self.esp32: Optional[ESP32] = None
        self.ml_engine: Any = None
        self.controller: Optional[IrrigationController] = None
        self.backend_thread: Optional[threading.Thread] = None

        self._last_display = 0.0

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self) -> None:

        self._print_header()

        # ----------------------------------------------------
        # 1. BACKEND
        # ----------------------------------------------------

        print()
        print("[1/5] Starting Flask backend...")

        self._start_backend()

        print("[OK] Flask backend started on 0.0.0.0:5000")

        # ----------------------------------------------------
        # 2. ESP32
        # ----------------------------------------------------

        print()
        print("[2/5] Connecting to ESP32...")

        self._initialize_esp32()

        if self.state.esp32_online:
            print("[OK] ESP32 connected.")
        else:
            print("[WARN] ESP32 unavailable; background polling will retry.")

        # ----------------------------------------------------
        # 3. ML
        # ----------------------------------------------------

        print()
        print("[3/5] Loading ML models...")

        self._initialize_ml()

        print("[OK] M1/M2/M3/M4 models loaded.")

        # ----------------------------------------------------
        # 4. WEATHER
        # ----------------------------------------------------

        print()
        print("[4/5] Initializing weather service...")

        self._initialize_weather()

        print("[OK] Weather service initialized.")

        # ----------------------------------------------------
        # 5. RUNTIME
        # ----------------------------------------------------

        print()
        print("[5/5] Starting runtime services...")

        self._start_threads()

        print()
        print("=" * 70)
        print("KRUSHI MITHRA SYSTEM RUNNING")
        print("=" * 70)

        if self.esp32 is not None:

            print(
                f"ESP32 Dashboard : "
                f"http://{self.esp32.ip}/"
            )

            print(
                f"ESP32 Setup     : "
                f"http://{self.esp32.ip}/wifi-setup"
            )

        print()
        print("Press CTRL+C to stop.")
        print("=" * 70)

    # ========================================================
    # ESP32 INITIALIZATION
    # ========================================================

    def _initialize_esp32(self) -> None:

        logger.info(
            "Creating ESP32 client: %s",
            ESP32_IP,
        )

        self.esp32 = ESP32()

        result = self.esp32.test_connection()

        if not result.get("connected"):
            logger.warning(
                "Initial ESP32 connection failed; runtime will retry in background: %s",
                result.get("error", "ESP32 connection failed."),
            )
            status = {}
        else:
            status = result["status"]
            logger.info("ESP32 connected successfully.")
            logger.info("ESP32 status: %s", status)

        # ----------------------------------------------------
        # Read initial data
        # ----------------------------------------------------

        data: Dict[str, Any] = {}
        config: Dict[str, Any] = {}
        if status:
            try:
                data = self.esp32.get_data()
            except ESP32Error as exc:
                logger.warning("Initial ESP32 sensor read failed: %s", exc)

            try:
                config = self.esp32.get_config()
            except ESP32Error as exc:
                logger.warning("Initial ESP32 config read failed: %s", exc)

        with self.state.lock:

            self.state.esp32_online = bool(status)
            self.state.esp32_status = status
            self.state.esp32_data = data
            self.state.esp32_config = config
        if data:
            update_cached_esp32_data(data)

        # ----------------------------------------------------
        # Controller
        # ----------------------------------------------------

        self.controller = IrrigationController(
            esp32=self.esp32,
        )

        # Synchronize control thresholds from ESP32.
        if status:
            try:
                self.controller.sync_config_from_esp32()
            except Exception as exc:
                logger.warning(
                    "Could not synchronize control config: %s",
                    exc,
                )

        self._print_runtime_state()

    # ========================================================
    # ML INITIALIZATION
    # ========================================================

    def _initialize_ml(self) -> None:

        logger.info(
            "Loading Krushi Mithra ML engine..."
        )

        # get_ml_engine() loads the models once
        # and keeps them in memory.
        self.ml_engine = get_ml_engine()

        logger.info(
            "M1/M2/M3/M4 models loaded successfully."
        )

    # ========================================================
    # WEATHER INITIALIZATION
    # ========================================================

    def _initialize_weather(self) -> None:

        try:
            history = M4WeatherHistory()
            if history.load().empty:
                logger.info(
                    "M4 weather history is empty. Queuing initial synchronization."
                )
                ensure_m4_history_ready(start_date_if_empty=RUNTIME_START_DATE)
            else:
                logger.info(
                    "M4 weather history already populated; skipping initial full sync."
                )

            weather = get_current_weather()

            with self.state.lock:
                self.state.weather = dict(weather)

            logger.info(
                "Weather updated."
            )

        except Exception as exc:

            logger.warning(
                "Initial weather update failed: %s",
                exc,
            )

            with self.state.lock:
                self.state.weather = {}

    def _start_backend(self) -> None:
        if self.backend_thread is not None and self.backend_thread.is_alive():
            return

        try:
            import requests
            response = requests.get("http://127.0.0.1:5000/api/health", timeout=1)
            if response.ok:
                logger.info("Existing Flask backend detected on port 5000; reusing it.")
                return
        except Exception:
            pass

        def run_backend() -> None:
            try:
                logger.info("Starting Flask backend on 0.0.0.0:5000")
                backend_app.run(
                    host="0.0.0.0",
                    port=5000,
                    debug=False,
                    threaded=True,
                    use_reloader=False,
                )
            except Exception:
                logger.exception("Flask backend failed to start.")

        self.backend_thread = threading.Thread(
            target=run_backend,
            name="FlaskBackendThread",
            daemon=True,
        )
        self.backend_thread.start()

        logger.info(
            "Flask backend thread started."
        )

    # ========================================================
    # THREAD CREATION
    # ========================================================

    def _start_threads(self) -> None:

        self._start_thread(
            target=self._esp32_sensor_loop,
            name="ESP32SensorLoop",
        )

        self._start_thread(
            target=self._weather_loop,
            name="WeatherLoop",
        )

        self._start_thread(
            target=self._control_loop,
            name="ControlLoop",
        )

        self._start_thread(
            target=self._display_loop,
            name="RuntimeDisplayLoop",
        )

    def _start_thread(
        self,
        target,
        name: str,
    ) -> None:

        thread = threading.Thread(
            target=target,
            name=name,
            daemon=True,
        )

        self.threads.append(thread)

        thread.start()

        logger.info(
            "Started thread: %s",
            name,
        )

    # ========================================================
    # ESP32 SENSOR LOOP
    # ========================================================

    def _esp32_sensor_loop(self) -> None:

        logger.info(
            "ESP32 sensor loop started."
        )

        failure_count = 0
        last_status_poll = time.monotonic() - ESP32_STATUS_POLL_INTERVAL

        while not self.stop_event.is_set():

            if self.esp32 is None:
                time.sleep(1)
                continue

            try:

                data = self.esp32.get_data()
                with self.state.lock:
                    self.state.esp32_data = dict(data)
                    self.state.esp32_online = True
                update_cached_esp32_data(data)
                if self.ml_engine is not None:
                    self.ml_engine.record_m3_reading(data)

                if failure_count:
                    logger.info("ESP32 telemetry recovered.")
                failure_count = 0

                now = time.monotonic()
                if now - last_status_poll >= ESP32_STATUS_POLL_INTERVAL:
                    last_status_poll = now
                    try:
                        status = self.esp32.get_status()
                    except ESP32Error as exc:
                        logger.warning("ESP32 status refresh failed: %s", exc)
                    else:
                        with self.state.lock:
                            self.state.esp32_status = dict(status)

            except ESP32Error as exc:

                with self.state.lock:
                    self.state.esp32_online = False

                failure_count += 1
                if failure_count == 1:
                    logger.warning("ESP32 connection lost: %s", exc)
                else:
                    logger.debug("ESP32 telemetry retry failed: %s", exc)

            except Exception as exc:

                logger.exception(
                    "Unexpected ESP32 polling error: %s",
                    exc,
                )

            base_interval = max(0.5, float(ESP32_POLL_INTERVAL))
            if failure_count:
                retry_delay = min(base_interval * (2 ** min(failure_count - 1, 4)), 15.0)
            else:
                retry_delay = base_interval
            self.stop_event.wait(retry_delay)

    # ========================================================
    # WEATHER LOOP
    # ========================================================

    def _weather_loop(self) -> None:

        logger.info(
            "Weather loop started."
        )

        last_m1_sync_date = None
        last_m4_sync_key = None

        while not self.stop_event.is_set():

            now = datetime.now()

            # ------------------------------------------------
            # Current weather
            # ------------------------------------------------

            try:

                weather = get_current_weather()

                with self.state.lock:
                    self.state.weather = dict(weather)

                logger.info(
                    "Weather updated."
                )

            except Exception as exc:

                logger.warning(
                    "Weather update failed: %s",
                    exc,
                )

            # ------------------------------------------------
            # M1 daily weather synchronization
            # ------------------------------------------------

            try:

                today = now.date()

                if (
                    last_m1_sync_date != today
                    and now.hour >= 1
                ):

                    logger.info(
                        "Synchronizing M1 weather history..."
                    )

                    added = sync_m1_weather(
                        start_date_if_empty=RUNTIME_START_DATE,
                    )

                    logger.info(
                        "M1 weather synchronization "
                        "completed. Added rows: %s",
                        added,
                    )

                    last_m1_sync_date = today

            except Exception as exc:

                logger.warning(
                    "M1 weather synchronization failed: %s",
                    exc,
                )

            # ------------------------------------------------
            # M4 twice-daily synchronization
            # ------------------------------------------------

            try:

                sync_key = (
                    now.date().isoformat(),
                    1 if now.hour >= M4_SYNC_HOUR_1 else 0,
                    2 if now.hour >= M4_SYNC_HOUR_2 else 0,
                )

                should_sync = (
                    now.hour in (
                        M4_SYNC_HOUR_1,
                        M4_SYNC_HOUR_2,
                    )
                    and last_m4_sync_key != sync_key
                )

                if should_sync:

                    logger.info(
                        "Queueing M4 weather synchronization in background..."
                    )

                    schedule_m4_sync(
                        start_date_if_empty=RUNTIME_START_DATE,
                    )

                    last_m4_sync_key = sync_key

            except Exception as exc:

                logger.warning(
                    "M4 weather synchronization failed: %s",
                    exc,
                )

            self.stop_event.wait(
                max(
                    10,
                    int(WEATHER_CURRENT_INTERVAL),
                )
            )

    # ========================================================
    # CONTROL LOOP
    # ========================================================

    def _control_loop(self) -> None:

        logger.info(
            "Control loop started."
        )

        last_heartbeat = 0.0

        while not self.stop_event.is_set():

            try:

                if self.controller is None:

                    logger.warning(
                        "Control loop waiting for controller."
                    )

                    self.stop_event.wait(1)
                    continue

                now = time.monotonic()
                if self.esp32 is not None and (now - last_heartbeat) >= float(BACKEND_HEARTBEAT_INTERVAL):
                    try:
                        self.esp32.send_heartbeat()
                        last_heartbeat = now
                        logger.debug("ESP32 heartbeat sent to refresh watchdog.")
                    except ESP32Error as exc:
                        logger.warning("ESP32 heartbeat failed: %s", exc)

                data = self.state.get_esp32_data()
                status = self.state.get_esp32_status()
                m2 = self.state.get_m2()
                m3 = self.state.get_m3()

                if not data or not status:

                    self.stop_event.wait(1)
                    continue

                # ------------------------------------------------
                # Only AUTO mode is acted upon here.
                #
                # In MANUAL mode the controller preserves
                # the existing pump state.
                # ------------------------------------------------

                result = self.controller.run_auto_cycle(
                    esp32_data=data,
                    status=status,
                    m2_result=m2,
                    m3_result=m3,
                )

                with self.state.lock:
                    self.state.control_result = dict(result)

            except ControlCommunicationError as exc:

                logger.warning(
                    "Control communication failed: %s",
                    exc,
                )

            except ControlError as exc:

                logger.warning(
                    "Control cycle failed: %s",
                    exc,
                )

            except ESP32Error as exc:

                logger.warning(
                    "ESP32 control error: %s",
                    exc,
                )

            except Exception as exc:

                logger.exception(
                    "Unexpected control loop error: %s",
                    exc,
                )

            # Avoid hammering ESP32.
            self.stop_event.wait(2)

    # ========================================================
    # RUNTIME DISPLAY
    # ========================================================

    def _display_loop(self) -> None:

        logger.info(
            "Runtime display loop started."
        )

        while not self.stop_event.is_set():

            try:

                self._print_runtime_state()

            except Exception as exc:

                logger.warning(
                    "Runtime display failed: %s",
                    exc,
                )

            self.stop_event.wait(30)

    # ========================================================
    # RUNTIME DISPLAY
    # ========================================================

    def _print_runtime_state(self) -> None:

        with self.state.lock:

            online = self.state.esp32_online

            status = dict(
                self.state.esp32_status
            )

            data = dict(
                self.state.esp32_data
            )

        print()
        print("-" * 70)
        print("KRUSHI MITHRA RUNTIME")
        print("-" * 70)

        print(
            f"ESP32: "
            f"{'ONLINE' if online else 'OFFLINE'}"
        )

        print(
            f"System : "
            f"{status.get('system_enabled', data.get('system_enabled', '-'))}"
        )

        print(
            f"Mode   : "
            f"{status.get('mode', data.get('mode', '-'))}"
        )

        print(
            f"Pump   : "
            f"{status.get('pump', data.get('pump', '-'))}"
        )

        print(
            f"IP     : "
            f"{status.get('ip', self.esp32.ip if self.esp32 else '-')}"
        )

        temperature = data.get(
            "temperature"
        )

        humidity = data.get(
            "humidity"
        )

        soil = data.get(
            "soil_moisture_percent"
        )

        soil_temperature = data.get(
            "soil_temperature"
        )

        rain = data.get(
            "rain_detected"
        )

        light = data.get(
            "light_detected"
        )

        if temperature is not None:
            print(
                f"Air    : {float(temperature):.1f} °C"
            )
        else:
            print("Air    : -")

        if humidity is not None:
            print(
                f"Humid  : {float(humidity):.1f} %"
            )
        else:
            print("Humid  : -")

        if soil is not None:
            print(
                f"Soil   : {float(soil):.1f} %"
            )
        else:
            print("Soil   : -")

        if soil_temperature is not None:
            print(
                f"Soil T : {float(soil_temperature):.1f} °C"
            )
        else:
            print("Soil T : -")

        print(
            f"Rain   : {rain if rain is not None else '-'}"
        )

        print(
            f"Light  : {light if light is not None else '-'}"
        )

        print("-" * 70)

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        if self.stop_event.is_set():
            return

        logger.info(
            "Shutting down Krushi Mithra..."
        )

        self.stop_event.set()

        with self.state.lock:
            self.state.running = False

        # ----------------------------------------------------
        # Safely turn pump OFF.
        # ----------------------------------------------------

        if self.controller is not None:

            try:

                self.controller.safe_pump_off(
                    reason="SYSTEM_SHUTDOWN"
                )

            except Exception as exc:

                logger.warning(
                    "Could not safely turn pump OFF: %s",
                    exc,
                )

        # ----------------------------------------------------
        # Wait for worker threads.
        # ----------------------------------------------------

        for thread in self.threads:

            if thread.is_alive():

                thread.join(
                    timeout=3
                )

        # ----------------------------------------------------
        # Close ESP32 session.
        # ----------------------------------------------------

        if self.esp32 is not None:

            try:
                self.esp32.close()

            except Exception:
                pass

        logger.info(
            "Krushi Mithra shutdown complete."
        )

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(self) -> None:

        try:

            self.initialize()

            while not self.stop_event.is_set():

                time.sleep(1)

        except KeyboardInterrupt:

            print()
            print(
                "\nCTRL+C received."
            )

        finally:

            self.shutdown()

    # ========================================================
    # HEADER
    # ========================================================

    @staticmethod
    def _print_header() -> None:

        print("=" * 70)
        print("KRUSHI MITHRA")
        print("SMART FARMING & INTELLIGENT IRRIGATION SYSTEM")
        print("=" * 70)

        print(
            f"System  : {SYSTEM_NAME}"
        )

        print(
            f"Version : {SYSTEM_VERSION}"
        )

        print(
            f"Timezone: {TIMEZONE}"
        )

        print(
            f"ESP32   : {ESP32_IP}"
        )


# ============================================================
# SIGNAL HANDLING
# ============================================================

_runtime: Optional[KrushiMithraRuntime] = None


def _handle_signal(
    signum,
    frame,
) -> None:

    global _runtime

    logger.info(
        "Received shutdown signal: %s",
        signum,
    )

    if _runtime is not None:
        _runtime.shutdown()


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    global _runtime

    setup_logging()

    signal.signal(
        signal.SIGINT,
        _handle_signal,
    )

    if hasattr(signal, "SIGTERM"):

        signal.signal(
            signal.SIGTERM,
            _handle_signal,
        )

    _runtime = KrushiMithraRuntime()

    _runtime.run()


if __name__ == "__main__":
    main()