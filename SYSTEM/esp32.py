"""
Krushi Mithra
Smart Farming & Intelligent Irrigation System

esp32.py
========

ESP32 communication layer.

Responsibilities:
- Communicate with ESP32 over local Wi-Fi/LAN.
- Read sensor data.
- Read ESP32 status.
- Read/update irrigation configuration.
- Control system state.
- Switch AUTO/MANUAL mode.
- Control pump.
- Handle Wi-Fi setup APIs.
- Test ESP32 connectivity.

The ESP32 itself handles:
- Wi-Fi connection.
- WiFiManager provisioning.
- Setup AP.
- Sensors.
- Relay GPIO23.
- Pump hardware.
- HTTP API endpoints.
- Final hardware safety.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import json
import logging
import socket
import threading
import time

import requests

from config import (
    ESP32_IP,
    ESP32_HOSTNAME,
    ESP32_TIMEOUT,
    ESP32_DISCOVERY_PORT,
    ESP32_DISCOVERY_MESSAGE,
    ESP32_DISCOVERY_TIMEOUT,
)


logger = logging.getLogger("KrushiMithra.ESP32")
_ESP32_REQUEST_LOCK = threading.RLock()
_DISCOVERY_RETRY_COOLDOWN = 30.0
_last_discovery_attempt = 0.0
_last_failure_log: Dict[str, float] = {}
_DISCOVERY_LAST_IP: Optional[str] = None


def resolve_esp32_ip(ip: Optional[str] = None) -> Optional[str]:
    """Resolve the ESP32 address in priority order: configured IP -> hostname -> discovery."""
    candidates: list[str] = []

    provided = ip if ip is not None else ESP32_IP
    cleaned = ESP32._clean_ip(provided) if provided is not None else None
    if cleaned:
        candidates.append(cleaned)

    if ESP32_HOSTNAME:
        candidates.append(str(ESP32_HOSTNAME))

    for candidate in candidates:
        try:
            host = candidate.strip()
            if not host or host.startswith("http"):
                continue
            if "." in host:
                return host
            resolved = socket.gethostbyname(host)
            if resolved:
                return resolved
        except OSError:
            pass

    discovered = discover_esp32()
    if discovered and discovered.get("ip"):
        return str(discovered["ip"])

    return None


# ============================================================
# EXCEPTIONS
# ============================================================

class ESP32Error(Exception):
    """Base exception for ESP32 communication errors."""


class ESP32ConnectionError(ESP32Error):
    """Raised when the ESP32 cannot be reached."""


class ESP32APIError(ESP32Error):
    """Raised when the ESP32 returns an API/HTTP error."""


def _candidate_broadcasts() -> list[str]:
    """Return broadcast targets that commonly work on Windows/Linux LANs."""
    targets = ["255.255.255.255"]

    # Also derive directed broadcasts from local IPv4 interfaces when
    # possible. This is useful on routers that do not forward the
    # limited broadcast address.
    try:
        import psutil  # optional; standard environments may not have it
        for _, addresses in psutil.net_if_addrs().items():
            ipv4 = None
            netmask = None
            for addr in addresses:
                if getattr(addr, "family", None) == socket.AF_INET:
                    ipv4 = addr.address
                    netmask = addr.netmask
                    break
            if not ipv4 or not netmask or ipv4.startswith("127."):
                continue
            try:
                import ipaddress
                network = ipaddress.IPv4Network(f"{ipv4}/{netmask}", strict=False)
                broadcast = str(network.broadcast_address)
                if broadcast not in targets:
                    targets.append(broadcast)
            except Exception:
                continue
    except Exception:
        pass

    return targets


def discover_esp32(
    timeout: Optional[float] = None,
    port: Optional[int] = None,
    message: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Discover a Krushi Mithra ESP32 on the local LAN without requiring
    a manually entered IP address.

    The ESP32 listens on UDP/4210 and replies with JSON containing its
    current LAN IP and hostname.
    """
    timeout = float(timeout or ESP32_DISCOVERY_TIMEOUT)
    port = int(port or ESP32_DISCOVERY_PORT)
    message = str(message or ESP32_DISCOVERY_MESSAGE)

    global _last_discovery_attempt
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)

    try:
        payload = message.encode("utf-8")
        for target in _candidate_broadcasts():
            try:
                sock.sendto(payload, (target, port))
            except OSError:
                continue

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            sock.settimeout(remaining)
            try:
                raw, addr = sock.recvfrom(1024)
            except socket.timeout:
                break
            except OSError:
                break

            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except (ValueError, UnicodeDecodeError):
                continue

            if not isinstance(data, dict):
                continue

            if data.get("device") != "KRUSHI_MITHRA":
                continue

            ip = str(data.get("ip") or addr[0]).strip()
            if not ip:
                continue

            data["ip"] = ip
            data.setdefault("port", 80)
            data.setdefault("hostname", "krushi-mithra.local")

            global _DISCOVERY_LAST_IP
            if _DISCOVERY_LAST_IP != ip:
                logger.info("ESP32 discovery resolved to %s", ip)
                _DISCOVERY_LAST_IP = ip
            return data

    finally:
        sock.close()

    return None


_DISCOVERY_LAST_IP = None


# ============================================================
# ESP32 CLIENT
# ============================================================

class ESP32:
    """
    Python client for the Krushi Mithra ESP32.

    Expected ESP32 endpoints:

        GET  /api/data
        GET  /api/status
        GET  /api/config

        POST /api/system
        POST /api/mode
        POST /api/pump
        POST /api/config

        GET  /api/wifi/scan
        POST /api/wifi/connect
        GET  /api/wifi/status
    """

    def __init__(
        self,
        ip: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.ip = self._clean_ip(ip or resolve_esp32_ip())

        self.timeout = (
            float(timeout)
            if timeout is not None
            else float(ESP32_TIMEOUT)
        )

        if not self.ip:
            self.ip = "krushi-mithra.local"

        self.base_url = f"http://{self.ip}"

        # Reuse one HTTP session.
        self.session = requests.Session()

        # Basic headers for ESP32 JSON API.
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    # ========================================================
    # INTERNAL HELPERS
    # ========================================================

    @staticmethod
    def _clean_ip(
        ip: Optional[str],
    ) -> Optional[str]:
        """
        Clean an ESP32 IP address or URL.

        Accepted examples:

            192.168.1.100
            http://192.168.1.100
            https://192.168.1.100/
        """

        if ip is None:
            return None

        ip = str(ip).strip()

        if not ip:
            return None

        if ip.startswith("http://"):
            ip = ip[7:]

        elif ip.startswith("https://"):
            ip = ip[8:]

        return ip.rstrip("/")

    def _url(
        self,
        endpoint: str,
    ) -> str:
        """Build a complete ESP32 API URL."""

        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        return f"{self.base_url}{endpoint}"

    def _set_ip(self, ip: str) -> None:
        """Switch targets and discard pooled sockets tied to the old IP."""
        cleaned_ip = self._clean_ip(ip)
        if not cleaned_ip:
            raise ValueError("Invalid ESP32 IP address.")
        if cleaned_ip == self.ip:
            return

        old_session = self.session
        self.ip = cleaned_ip
        self.base_url = f"http://{cleaned_ip}"
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        old_session.close()

    # ========================================================
    # REQUEST HANDLING
    # ========================================================

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP request and return JSON.

        IMPORTANT:
        POST data is sent using JSON, not form encoding.
        """

        global _last_discovery_attempt

        method = method.upper()
        request_timeout = float(timeout) if timeout is not None else self.timeout
        timeout_pair = (min(5.0, request_timeout), request_timeout)

        def perform_request(request_url: str):
            if method == "GET":
                return self.session.get(request_url, timeout=timeout_pair)
            return self.session.request(
                method=method,
                url=request_url,
                json=data or {},
                timeout=timeout_pair,
            )

        # The ESP32 uses a synchronous WebServer; serialize every Python
        # client in this process so backend and runtime calls cannot overlap.
        with _ESP32_REQUEST_LOCK:
            started = time.monotonic()
            url = self._url(endpoint)
            logger.debug("ESP32 %s %s started", method, endpoint)

            try:
                try:
                    response = perform_request(url)
                except requests.RequestException as first_exc:
                    now = time.monotonic()
                    discovered = None
                    if now - _last_discovery_attempt >= _DISCOVERY_RETRY_COOLDOWN:
                        _last_discovery_attempt = now
                        discovered = discover_esp32()

                    discovered_ip = discovered.get("ip") if discovered else None
                    if discovered_ip and discovered_ip != self.ip:
                        old_ip = self.ip
                        self._set_ip(str(discovered_ip))
                        url = self._url(endpoint)
                        logger.info(
                            "ESP32 rediscovered at %s (previous target %s)",
                            self.ip,
                            old_ip,
                        )
                        try:
                            response = perform_request(url)
                        except requests.RequestException as retry_exc:
                            raise ESP32ConnectionError(
                                f"ESP32 unreachable after discovery at {url}: {retry_exc}"
                            ) from retry_exc
                    else:
                        raise ESP32ConnectionError(
                            f"ESP32 unreachable at {url}: {first_exc}"
                        ) from first_exc

                if not response.ok:
                    raise ESP32APIError(
                        f"ESP32 returned HTTP {response.status_code} for "
                        f"{method} {endpoint}: {response.text[:500]}"
                    )

                try:
                    result = response.json()
                except ValueError as exc:
                    raise ESP32APIError(
                        f"ESP32 returned invalid JSON for {method} {endpoint}: "
                        f"{response.text[:500]}"
                    ) from exc

                if not isinstance(result, dict):
                    raise ESP32APIError(
                        f"ESP32 returned unexpected JSON type for {method} {endpoint}."
                    )

                logger.debug(
                    "ESP32 %s %s completed in %.3fs",
                    method,
                    endpoint,
                    time.monotonic() - started,
                )
                _last_failure_log.pop(endpoint, None)
                return result
            except (ESP32ConnectionError, ESP32APIError) as exc:
                now = time.monotonic()
                previous_log = _last_failure_log.get(endpoint, 0.0)
                if now - previous_log >= 30.0:
                    _last_failure_log[endpoint] = now
                    logger.warning(
                        "ESP32 request failed: endpoint=%s error=%s",
                        endpoint,
                        exc,
                    )
                else:
                    logger.debug(
                        "ESP32 request retry failed: endpoint=%s error=%s",
                        endpoint,
                        exc,
                    )
                raise

    def _get(
        self,
        endpoint: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform GET request."""

        return self._request(
            method="GET",
            endpoint=endpoint,
            timeout=timeout,
        )

    def _post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform JSON POST request."""

        return self._request(
            method="POST",
            endpoint=endpoint,
            data=data or {},
            timeout=timeout,
        )

    # ========================================================
    # SENSOR DATA
    # ========================================================

    def get_data(self) -> Dict[str, Any]:
        """
        Get current ESP32 sensor data.

        Expected:

        {
            "temperature": 28.8,
            "humidity": 65.9,
            "soil_moisture_raw": 2802,
            "soil_moisture_percent": 49.8,
            "soil_temperature": 29.3,
            "rain_detected": 0,
            "rain_wetness_raw": 4095,
            "light_detected": 1,
            "pump": false,
            "system_enabled": true,
            "mode": "AUTO"
        }
        """

        return self._get("/api/data")

    # ========================================================
    # ESP32 STATUS
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get ESP32/system status.

        Expected:

        {
            "online": true,
            "system_enabled": true,
            "mode": "AUTO",
            "pump": false,
            "pump_source": "none",
            "wifi_connected": true,
            "wifi_ssid": "...",
            "wifi_rssi": -50,
            "ip": "192.168.x.x"
        }
        """

        return self._get("/api/status")

    # ========================================================
    # ESP32 CONFIGURATION
    # ========================================================

    def get_config(self) -> Dict[str, Any]:
        """
        Get irrigation/system configuration.
        """

        return self._get("/api/config")

    # ========================================================
    # SYSTEM ON/OFF
    # ========================================================

    def set_system(
        self,
        enabled: bool,
    ) -> Dict[str, Any]:
        """
        Enable or disable Krushi Mithra automation.

        SYSTEM OFF:
            - Automation disabled.
            - Pump forced OFF by ESP32.
            - Dashboard/API remain available.

        SYSTEM ON:
            - Automation is allowed.
        """

        payload = {
            "enabled": bool(enabled),
        }

        return self._post(
            "/api/system",
            payload,
        )

    def enable_system(self) -> Dict[str, Any]:
        """Enable Krushi Mithra."""

        return self.set_system(True)

    def disable_system(self) -> Dict[str, Any]:
        """Disable Krushi Mithra."""

        return self.set_system(False)

    # ========================================================
    # MODE
    # ========================================================

    def set_mode(
        self,
        mode: str,
    ) -> Dict[str, Any]:
        """
        Set ESP32 operating mode.

        Valid:
            AUTO
            MANUAL
        """

        mode = str(mode).strip().upper()

        if mode not in ("AUTO", "MANUAL"):

            raise ValueError(
                "Mode must be either 'AUTO' or 'MANUAL'."
            )

        return self._post(
            "/api/mode",
            {
                "mode": mode,
            },
        )

    def set_auto(self) -> Dict[str, Any]:
        """Switch ESP32 to AUTO mode."""

        return self.set_mode("AUTO")

    def set_manual(self) -> Dict[str, Any]:
        """Switch ESP32 to MANUAL mode."""

        return self.set_mode("MANUAL")

    # ========================================================
    # PUMP CONTROL
    # ========================================================

    def set_pump(
        self,
        state: bool,
        source: str = "pc",
        duration_value: Optional[float] = None,
        duration_unit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Request pump ON/OFF.

        source determines who is requesting the command:

            "pc"
                Python AUTO controller.

            "manual"
                Dashboard/manual user control.

        The ESP32 remains the final safety authority.

        Expected JSON:

            {
                "state": true,
                "source": "pc"
            }

        or:

            {
                "state": true,
                "source": "manual"
            }
        """

        source = str(source).strip().lower()

        if source not in (
            "pc",
            "manual",
        ):
            raise ValueError(
                "Pump source must be 'pc' or 'manual'."
            )

        payload: Dict[str, Any] = {
            "state": bool(state),
            "source": source,
        }

        if state and duration_value is not None:
            value = float(duration_value)
            if value <= 0:
                raise ValueError("duration_value must be greater than 0.")
            payload["duration_value"] = value

            unit = str(duration_unit or "SEC").strip().upper()
            if unit not in ("SEC", "MIN", "HR"):
                raise ValueError("duration_unit must be SEC, MIN, or HR.")
            payload["duration_unit"] = unit

        return self._post(
            "/api/pump",
            payload,
        )

    def pump_on(
        self,
        source: str = "pc",
        duration_value: Optional[float] = None,
        duration_unit: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Turn pump ON, optionally using ESP32 timed-manual control."""

        return self.set_pump(
            True,
            source=source,
            duration_value=duration_value,
            duration_unit=duration_unit,
        )

    def pump_off(
        self,
        source: str = "pc",
    ) -> Dict[str, Any]:
        """Turn pump OFF using the specified source."""

        return self.set_pump(
            False,
            source=source,
        )

    # ========================================================
    # IRRIGATION CONFIGURATION
    # ========================================================

    def set_config(
        self,
        soil_trigger: Optional[float] = None,
        critical_soil: Optional[float] = None,
        rain_protection: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Update irrigation configuration.

        Only supplied values are changed.
        """

        payload: Dict[str, Any] = {}

        # ----------------------------------------------------
        # SOIL TRIGGER
        # ----------------------------------------------------

        if soil_trigger is not None:

            soil_trigger = float(soil_trigger)

            if not 0 <= soil_trigger <= 100:

                raise ValueError(
                    "soil_trigger must be between 0 and 100."
                )

            payload["soil_trigger"] = soil_trigger

        # ----------------------------------------------------
        # CRITICAL SOIL
        # ----------------------------------------------------

        if critical_soil is not None:

            critical_soil = float(critical_soil)

            if not 0 <= critical_soil <= 100:

                raise ValueError(
                    "critical_soil must be between 0 and 100."
                )

            payload["critical_soil"] = critical_soil

        # ----------------------------------------------------
        # RAIN PROTECTION
        # ----------------------------------------------------

        if rain_protection is not None:

            payload["rain_protection"] = bool(
                rain_protection
            )

        # Nothing to update.
        if not payload:
            return self.get_config()

        return self._post(
            "/api/config",
            payload,
        )

    # ========================================================
    # WIFI
    # ========================================================

    def scan_wifi(
        self,
    ) -> Dict[str, Any]:
        """
        Ask ESP32 to scan nearby Wi-Fi networks.

        This is handled by the ESP32.
        Python does not scan the PC's Wi-Fi adapter.
        """

        return self._get(
            "/api/wifi/scan",
            timeout=max(self.timeout, 15.0),
        )

    def connect_wifi(
        self,
        ssid: str,
        password: str,
    ) -> Dict[str, Any]:
        """
        Request ESP32 to connect to a Wi-Fi network.

        Wi-Fi credentials are sent to ESP32 only.
        They are not stored by this Python client.
        """

        ssid = str(ssid).strip()

        if not ssid:
            raise ValueError(
                "Wi-Fi SSID cannot be empty."
            )

        payload = {
            "ssid": ssid,
            "password": str(password),
        }

        return self._post(
            "/api/wifi/connect",
            payload,
            timeout=max(self.timeout, 15.0),
        )

    def get_wifi_status(
        self,
    ) -> Dict[str, Any]:
        """Get ESP32 Wi-Fi status."""

        return self._get(
            "/api/wifi/status"
        )

    # ========================================================
    # DISCOVERY
    # ========================================================

    def send_heartbeat(self) -> Dict[str, Any]:
        """Ping the ESP32 to refresh the backend watchdog timer."""
        return self._post("/api/backend/heartbeat", data={})

    @classmethod
    def discover(cls, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Discover the ESP32 on the local LAN."""
        return discover_esp32(timeout=timeout)

    # ========================================================
    # CONNECTION TEST
    # ========================================================

    def test_connection(
        self,
    ) -> Dict[str, Any]:
        """
        Test communication with ESP32.

        Returns:

        {
            "connected": True,
            "status": {...}
        }

        or:

        {
            "connected": False,
            "error": "..."
        }
        """

        try:

            status = self.get_status()

            return {
                "connected": True,
                "status": status,
            }

        except ESP32Error as exc:

            return {
                "connected": False,
                "error": str(exc),
            }

    # ========================================================
    # CLOSE SESSION
    # ========================================================

    def close(self) -> None:
        """Close the HTTP session."""
        with _ESP32_REQUEST_LOCK:
            self.session.close()

    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            f"ESP32("
            f"ip='{self.ip}', "
            f"timeout={self.timeout}"
            f")"
        )


# ================================================================
# SIMPLE ESP32 TEST
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("KRUSHI MITHRA - ESP32 COMMUNICATION TEST")
    print("=" * 70)

    esp32 = None

    try:

        esp32 = ESP32()

        print()
        print(f"ESP32: {esp32}")
        print()

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        print("[1] STATUS")

        status = esp32.get_status()

        print(status)

        # ----------------------------------------------------
        # SENSOR DATA
        # ----------------------------------------------------

        print()
        print("[2] SENSOR DATA")

        data = esp32.get_data()

        print(data)

        # ----------------------------------------------------
        # CONFIG
        # ----------------------------------------------------

        print()
        print("[3] CONFIG")

        config = esp32.get_config()

        print(config)

        # ----------------------------------------------------
        # CONNECTION
        # ----------------------------------------------------

        print()
        print("[4] CONNECTION TEST")

        result = esp32.test_connection()

        print(result)

        if result["connected"]:
            print()
            print("=" * 70)
            print("ESP32 COMMUNICATION TEST PASSED")
            print("=" * 70)

        else:
            print()
            print("=" * 70)
            print("ESP32 COMMUNICATION TEST FAILED")
            print("=" * 70)

    except Exception as exc:

        print()
        print("=" * 70)
        print("ESP32 COMMUNICATION TEST FAILED")
        print("=" * 70)
        print()
        print(f"ERROR: {exc}")

    finally:

        if esp32 is not None:
            esp32.close()