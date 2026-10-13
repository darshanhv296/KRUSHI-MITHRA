"""
Krushi Mithra
Smart Farming & Intelligent Irrigation System

backend.py
==========
Flask backend connecting:

Browser UI
    -> backend.py
    -> esp32.py
    -> ESP32

and

Browser UI
    -> backend.py
    -> ml.py
    -> M1 / M2 / M3 / M4

The ESP32 remains responsible for:
- Wi-Fi / WiFiManager provisioning
- sensors
- relay GPIO23
- pump hardware
- ESP32-side safety rules

This backend does NOT store Wi-Fi passwords.
"""

from __future__ import annotations

import logging
import ipaddress
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from flask import Flask, jsonify, request, send_file
from zoneinfo import ZoneInfo

from config import (
    ESP32_IP,
    ESP32_TIMEOUT,
    SYSTEM_NAME,
    SYSTEM_VERSION,
    M1_LOCATION,
    ESP32_POLL_INTERVAL,
    WEATHER_LATITUDE,
    WEATHER_LONGITUDE,
    WEATHER_TIMEZONE,
)

from esp32 import (
    ESP32,
    ESP32Error,
    ESP32ConnectionError,
    ESP32APIError,
)
from weather import (
    M1_ALLOWED_LOCATIONS,
    M1_LOCATION_COORDINATES,
    PRIMARY_REGION,
    get_current_weather_for_location,
    get_current_weather_at,
    REGIONS,
    get_m1_annual_rainfall_reference,
    get_m3_rain_sum_6h,
    normalize_location_name,
    normalize_weather_region,
)

# ------------------------------------------------------------
# APP
# ------------------------------------------------------------

app = Flask(__name__)
_SENSOR_CACHE_LOCK = threading.RLock()
_LATEST_ESP32_DATA: Dict[str, Any] = {}
_LATEST_ESP32_DATA_UPDATED_AT: Optional[float] = None
M3_SENSOR_CACHE_MAX_AGE_SECONDS = max(10.0, float(ESP32_POLL_INTERVAL) * 5)

SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent

# Optional Wi-Fi setup UI; the ESP32 serves the production dashboard.
WIFI_SETUP_CANDIDATES = [
    SYSTEM_DIR / "44.html",
    PROJECT_ROOT / "44.html",
    SYSTEM_DIR / "wifi_setup.html",
    PROJECT_ROOT / "wifi_setup.html",
]

SETUP_AP_IP = "192.168.4.1"

logger = logging.getLogger("KrushiMithra.Backend")
M1_IRRIGATION_TYPES = ("Drip", "Sprinkler", "Rainfed", "Canal")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@app.after_request
def log_ml_response(response):
    if request.path.startswith("/api/ml/"):
        logger.info(
            "ML response: endpoint=%s status=%s response_json=%s",
            request.path,
            response.status_code,
            response.get_json(silent=True),
        )
    return response




def update_cached_esp32_data(data: Dict[str, Any]) -> None:
    """Publish the latest runtime telemetry for ML requests proxied by the ESP32."""
    global _LATEST_ESP32_DATA_UPDATED_AT
    with _SENSOR_CACHE_LOCK:
        _LATEST_ESP32_DATA.clear()
        _LATEST_ESP32_DATA.update(data)
        _LATEST_ESP32_DATA_UPDATED_AT = time.monotonic()


def get_cached_esp32_data() -> Dict[str, Any]:
    with _SENSOR_CACHE_LOCK:
        return dict(_LATEST_ESP32_DATA)


def get_cached_esp32_data_age_seconds() -> Optional[float]:
    with _SENSOR_CACHE_LOCK:
        if _LATEST_ESP32_DATA_UPDATED_AT is None:
            return None
        return max(0.0, time.monotonic() - _LATEST_ESP32_DATA_UPDATED_AT)


def get_ml_sensor_data(body: Dict[str, Any]) -> Dict[str, Any]:
    """Build ML sensor input from cached telemetry plus request overrides."""
    return {**get_cached_esp32_data(), **body}


def has_complete_m3_sensor_reading(data: Dict[str, Any]) -> bool:
    required_aliases = (
        ("soil_moisture_pct", "soil_moisture_percent", "soil_moisture_raw"),
        ("air_temperature_c", "temperature"),
        ("air_humidity_pct", "humidity"),
        ("soil_temperature_c", "soil_temperature"),
    )
    return all(any(data.get(name) is not None for name in aliases) for aliases in required_aliases)


# ------------------------------------------------------------
# ESP32 CLIENT MANAGER
# ------------------------------------------------------------

class ESP32Manager:
    """
    Keeps one reusable ESP32 communication object.

    ESP32_IP can initially be supplied through config.py.
    If it is not configured, Wi-Fi setup endpoints can still
    talk to the ESP32 setup AP at 192.168.4.1.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._ip: Optional[str] = self._clean_ip(ESP32_IP)
        self._client: Optional[ESP32] = None

        if self._ip:
            self._client = ESP32(
                ip=self._ip,
                timeout=ESP32_TIMEOUT,
            )

    @staticmethod
    def _clean_ip(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        value = value.replace("http://", "")
        value = value.replace("https://", "")
        return value.rstrip("/")

    @property
    def ip(self) -> Optional[str]:
        with self._lock:
            if self._client is not None and self._client.ip:
                return self._client.ip
            return self._ip

    def set_ip(self, ip: str) -> str:
        cleaned = self._clean_ip(ip)

        if not cleaned:
            raise ValueError("Invalid ESP32 IP address.")

        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass

            self._ip = cleaned
            self._client = ESP32(
                ip=cleaned,
                timeout=ESP32_TIMEOUT,
            )

        logger.info("ESP32 IP updated to %s", cleaned)
        return cleaned

    def discover(self) -> Optional[str]:
        """Discover the ESP32 on the local LAN and update this manager."""
        discovered = ESP32.discover()
        if not discovered or not discovered.get("ip"):
            return None

        ip = self.set_ip(str(discovered["ip"]))
        logger.info(
            "ESP32 discovered automatically: %s (%s)",
            ip,
            discovered.get("hostname", "krushi-mithra.local"),
        )
        return ip

    def require_client(self) -> ESP32:
        with self._lock:
            client = self._client

        if client is not None:
            return client

        discovered_ip = self.discover()
        if discovered_ip:
            with self._lock:
                if self._client is not None:
                    return self._client

        raise ESP32ConnectionError(
            "ESP32 could not be discovered on the local LAN. "
            "Make sure the ESP32 is connected to the same Wi-Fi network "
            "as this PC."
        )

    def status(self) -> Dict[str, Any]:
        client = self.require_client()
        result = client.get_status()

        # If ESP32 reports its actual LAN IP, keep backend aligned.
        reported_ip = result.get("ip")
        if reported_ip:
            reported_ip = self._clean_ip(str(reported_ip))
            if reported_ip and reported_ip != self.ip:
                try:
                    self.set_ip(reported_ip)
                except Exception as exc:
                    logger.warning(
                        "Could not update ESP32 IP from status: %s",
                        exc,
                    )

        return result


esp32_manager = ESP32Manager()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def json_error(message: str, status: int = 400):
    return jsonify({
        "success": False,
        "error": message,
    }), status


def get_json_body() -> Dict[str, Any]:
    body = request.get_json(silent=True)

    if body is None:
        return {}

    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")

    return body


def require_value(
    data: Dict[str, Any],
    name: str,
) -> Any:
    value = data.get(name)

    if value is None:
        raise ValueError(f"Missing required field: {name}")

    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Field cannot be empty: {name}")

    return value


def as_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

    if text in ("true", "1", "yes", "on"):
        return True

    if text in ("false", "0", "no", "off"):
        return False

    raise ValueError(f"Invalid boolean value: {value}")


def normalize_m1_location(location: Any) -> str:
    try:
        return normalize_location_name(location)
    except ValueError:
        raise


def get_m1_weather_for_location(location: Any) -> Dict[str, Any]:
    normalized = normalize_m1_location(location)

    try:
        weather = get_current_weather_for_location(normalized)
    except Exception as exc:
        message = str(exc)
        if "Unsupported M1 location" in message:
            raise ValueError(message) from exc
        raise RuntimeError(
            f"Weather data unavailable for selected location: {normalized}."
        ) from exc

    if weather.get("temperature_c") is None or weather.get("humidity_percent") is None:
        raise RuntimeError(
            f"Weather data unavailable for selected location: {normalized}."
        )

    return weather


def get_esp32_data() -> Dict[str, Any]:
    return esp32_manager.require_client().get_data()


def get_current_weather_safe() -> Dict[str, Any]:
    """
    Load the current weather provider only when an ML endpoint
    actually needs it.
    """
    from weather import get_current_weather

    weather = get_current_weather()

    if not isinstance(weather, dict):
        raise RuntimeError("Weather provider returned invalid data.")

    return weather


def get_ml_engine():
    """
    Lazy-load the ML engine.

    This keeps backend.py startup independent from ML model loading.
    The four models are still loaded once by ml.py when first used.
    """
    from ml import get_ml_engine
    return get_ml_engine()


def normalize_m3_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return both public M3 value aliases as finite floats and pin its model label."""
    value = None
    for key in ("future_soil_moisture_percent", "target_moisture_1h"):
        raw_value = result.get(key)
        if raw_value is None:
            continue
        try:
            candidate = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            value = candidate
            break

    if value is None:
        raise RuntimeError("M3 model returned no finite soil-moisture prediction.")

    return {
        **result,
        "success": True,
        "model": "LightGBM",
        "target_moisture_1h": value,
        "future_soil_moisture_percent": value,
    }


def find_ui_file(candidates):
    for path in candidates:
        if path.exists() and path.is_file():
            return path

    return None


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    """
    Allows the UI to be opened directly from ESP32 or another
    local development server while the Python backend runs on PC.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization"
    )
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )
    return response


# ------------------------------------------------------------
# UI ROUTES
# ------------------------------------------------------------

@app.route("/wifi-setup", methods=["GET"])
def wifi_setup():
    path = find_ui_file(WIFI_SETUP_CANDIDATES)

    if path is None:
        return jsonify({
            "success": False,
            "error": (
                "Wi-Fi setup HTML not found. Put 44.html or "
                "wifi_setup.html beside backend.py."
            ),
        }), 404

    return send_file(path)




# ------------------------------------------------------------
# BACKEND STATUS
# ------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "success": True,
        "backend": "online",
        "system": SYSTEM_NAME,
        "version": SYSTEM_VERSION,
    })


@app.route("/api/backend/status", methods=["GET"])
def backend_status():
    connected = False
    status: Dict[str, Any] = {}
    error = None

    try:
        status = esp32_manager.status()
        connected = True
    except Exception as exc:
        error = str(exc)

    return jsonify({
        "success": True,
        "backend": "online",
        "esp32": {
            "connected": connected,
            "ip": esp32_manager.ip,
            "status": status,
            "error": error,
        },
    })


@app.route("/api/device/discover", methods=["GET", "POST"])
def device_discover():
    try:
        ip = esp32_manager.discover()
        if not ip:
            return json_error(
                "Krushi Mithra ESP32 was not found on the local network.",
                503,
            )

        status = esp32_manager.status()
        return jsonify({
            "success": True,
            "ip": ip,
            "status": status,
        })

    except ESP32Error as exc:
        return json_error(str(exc), 503)
    except Exception as exc:
        logger.exception("ESP32 discovery failed")
        return json_error(str(exc), 500)


@app.route("/api/device/status", methods=["GET"])
def device_status():
    try:
        status = esp32_manager.status()

        return jsonify({
            "success": True,
            "esp32": status,
        })

    except ESP32Error as exc:
        return json_error(str(exc), 503)


@app.route("/api/device/ip", methods=["POST"])
def set_device_ip():
    try:
        body = get_json_body()
        ip = require_value(body, "ip")

        new_ip = esp32_manager.set_ip(str(ip))

        return jsonify({
            "success": True,
            "ip": new_ip,
        })

    except ValueError as exc:
        return json_error(str(exc), 400)

    except Exception as exc:
        logger.exception("Failed to set ESP32 IP")
        return json_error(str(exc), 500)


# ------------------------------------------------------------
# ESP32 DATA
# ------------------------------------------------------------

@app.route("/api/data", methods=["GET"])
def api_data():
    try:
        data = get_esp32_data()
        update_cached_esp32_data(data)
        return jsonify(data)

    except ESP32Error as exc:
        return json_error(str(exc), 503)


@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        status = esp32_manager.status()

        return jsonify(status)

    except ESP32Error as exc:
        return json_error(str(exc), 503)


@app.route("/api/config", methods=["GET"])
def api_config_get():
    try:
        config = esp32_manager.require_client().get_config()

        return jsonify(config)

    except ESP32Error as exc:
        return json_error(str(exc), 503)


# ------------------------------------------------------------
# ESP32 SYSTEM CONTROL
# ------------------------------------------------------------

@app.route("/api/system", methods=["POST"])
def api_system():
    try:
        body = get_json_body()
        enabled = as_bool(require_value(body, "enabled"))

        result = esp32_manager.require_client().set_system(
            bool(enabled)
        )

        return jsonify(result)

    except ValueError as exc:
        return json_error(str(exc), 400)

    except ESP32Error as exc:
        return json_error(str(exc), 503)


@app.route("/api/mode", methods=["POST"])
def api_mode():
    try:
        body = get_json_body()
        mode = str(require_value(body, "mode")).strip().upper()

        if mode not in ("AUTO", "MANUAL"):
            return json_error(
                "mode must be AUTO or MANUAL.",
                400,
            )

        result = esp32_manager.require_client().set_mode(mode)

        return jsonify(result)

    except ValueError as exc:
        return json_error(str(exc), 400)

    except ESP32Error as exc:
        return json_error(str(exc), 503)


@app.route("/api/pump", methods=["POST"])
def api_pump():
    try:
        body = get_json_body()
        state = as_bool(require_value(body, "state"))

        source = str(body.get("source") or "pc").strip().lower()
        if source not in ("pc", "manual"):
            return json_error(
                "source must be 'pc' or 'manual'.",
                400,
            )

        duration_value = body.get("duration_value")
        duration_unit = body.get("duration_unit")

        if duration_value is not None:
            duration_value = float(duration_value)
            if duration_value <= 0:
                return json_error(
                    "duration_value must be greater than 0.",
                    400,
                )

        result = esp32_manager.require_client().set_pump(
            bool(state),
            source=source,
            duration_value=duration_value,
            duration_unit=duration_unit,
        )

        return jsonify(result)

    except ValueError as exc:
        return json_error(str(exc), 400)

    except ESP32Error as exc:
        return json_error(str(exc), 503)


@app.route("/api/config", methods=["POST"])
def api_config_post():
    try:
        body = get_json_body()

        soil_trigger = body.get("soil_trigger")
        critical_soil = body.get("critical_soil")
        rain_protection = body.get("rain_protection")

        if soil_trigger is not None:
            soil_trigger = float(soil_trigger)

        if critical_soil is not None:
            critical_soil = float(critical_soil)

        if rain_protection is not None:
            rain_protection = as_bool(rain_protection)

        result = esp32_manager.require_client().set_config(
            soil_trigger=soil_trigger,
            critical_soil=critical_soil,
            rain_protection=rain_protection,
        )

        return jsonify(result)

    except ValueError as exc:
        return json_error(str(exc), 400)

    except ESP32Error as exc:
        return json_error(str(exc), 503)


# ------------------------------------------------------------
# M1 - CROP RECOMMENDATION
# ------------------------------------------------------------

@app.route("/api/ml/crop", methods=["POST"])
def api_ml_crop():
    """M1 crop recommendation using request payload + weather context."""

    try:
        body = get_json_body()

        location = normalize_m1_location(body.get("location"))
        soil_type = str(require_value(body, "soil_type")).strip()
        irrigation = str(require_value(body, "irrigation")).strip()

        if not soil_type:
            raise ValueError("Soil type is required.")
        if not irrigation:
            raise ValueError("Irrigation type is required.")

        engine = get_ml_engine()
        prediction_options = engine.prediction_options()
        if soil_type not in prediction_options["soil_types"]:
            raise ValueError(
                f"Unsupported soil type: {soil_type}. "
                f"Choose one of: {', '.join(prediction_options['soil_types'])}."
            )

        weather = get_m1_weather_for_location(location)
        reference_date = datetime.now(ZoneInfo("Asia/Kolkata"))

        request_temperature = body.get("temperature")
        request_humidity = body.get("humidity")
        soil_moisture = body.get("soil_moisture")
        sensor_data = get_cached_esp32_data()

        if soil_moisture is None and body.get("soil_moisture_percent") is not None:
            soil_moisture = body.get("soil_moisture_percent")
        if soil_moisture is None and body.get("soil_moisture_pct") is not None:
            soil_moisture = body.get("soil_moisture_pct")
        if soil_moisture is None:
            soil_moisture = sensor_data.get(
                "soil_moisture_percent",
                sensor_data.get("soil_moisture_pct"),
            )

        temperature = request_temperature
        if temperature is None:
            temperature = body.get("air_temperature_c")
        if temperature is None:
            temperature = sensor_data.get(
                "temperature",
                sensor_data.get("air_temperature_c"),
            )
        if temperature is None:
            temperature = weather.get("temperature_c")
        humidity = request_humidity
        if humidity is None:
            humidity = body.get("air_humidity_percent", body.get("air_humidity_pct"))
        if humidity is None:
            humidity = sensor_data.get(
                "humidity",
                sensor_data.get("air_humidity_percent", sensor_data.get("air_humidity_pct")),
            )
        if humidity is None:
            humidity = weather.get("humidity_percent")

        if temperature is None:
            raise ValueError("Temperature is required for M1 crop recommendation.")
        if humidity is None:
            raise ValueError("Humidity is required for M1 crop recommendation.")
        if soil_moisture is None:
            raw_soil_moisture = body.get(
                "soil_moisture_raw",
                sensor_data.get("soil_moisture_raw"),
            )
            if raw_soil_moisture is not None:
                from ml import get_ml_engine as load_ml_engine

                soil_moisture = load_ml_engine().soil_raw_to_percent(float(raw_soil_moisture))
        if soil_moisture is None:
            raise ValueError(
                "Soil moisture is unavailable: ESP32 telemetry has not reached the Python runtime yet."
            )

        annual_rainfall_mm = body.get("annual_rainfall_mm")
        if annual_rainfall_mm is None:
            annual_rainfall_mm = weather.get("annual_rainfall_mm")

        if annual_rainfall_mm is None:
            annual_rainfall_mm = get_m1_annual_rainfall_reference(location)

        result = engine.predict_m1(
            location=location,
            soil_type=soil_type,
            irrigation=irrigation,
            temperature=float(temperature),
            humidity=float(humidity),
            soil_moisture=float(soil_moisture),
            annual_rainfall_mm=float(annual_rainfall_mm) if annual_rainfall_mm is not None else None,
            reference_date=reference_date,
            region=location,
        )
        if not result.get("success"):
            return jsonify(result), 500

        response = {
            "success": True,
            "location": location,
            "weather_source": "Open-Meteo",
            "weather_temperature": float(temperature),
            "weather_humidity": float(humidity),
            "weather_rainfall_mm": float(weather.get("rainfall_mm", 0.0)),
            **result,
        }

        return jsonify(response)

    except ValueError as exc:
        logger.warning("M1 validation failed for %s: %s", request.path, exc)
        return json_error(str(exc), 400)

    except RuntimeError as exc:
        logger.warning("M1 weather lookup failed for %s: %s", request.path, exc)
        return json_error(str(exc), 503)

    except ESP32Error as exc:
        logger.warning("M1 request reached ESP32 dependency unexpectedly for %s: %s", request.path, exc)
        return json_error(str(exc), 503)

    except Exception as exc:
        logger.exception("M1 crop prediction failed for %s", request.path)
        return json_error(str(exc), 500)


@app.route("/api/ml/options", methods=["GET"])
def api_ml_options():
    try:
        options = get_ml_engine().prediction_options()
        if set(options["crop_locations"]) != set(options["rainfall_regions"]):
            raise RuntimeError(
                "M1 crop locations and M4 rainfall regions do not match the loaded models."
            )

        return jsonify({
            "success": True,
            **options,
            "irrigation_types": list(M1_IRRIGATION_TYPES),
        })
    except Exception as exc:
        logger.exception("Could not load trained prediction options")
        return json_error(f"Prediction options unavailable: {exc}", 503)


# ------------------------------------------------------------
# M2 - CONTEXT-AWARE IRRIGATION
# ------------------------------------------------------------

@app.route("/api/ml/irrigation", methods=["POST"])
def api_ml_irrigation():
    """
    Optional direct M2 endpoint.

    Payload:

    {
        "crop_type": "Rice",
        "best_time": "Morning"
    }

    Sensor values come from ESP32.
    """

    try:
        body = get_json_body()

        crop_type = str(require_value(body, "crop_type"))
        best_time = str(body.get("best_time", "Any"))

        data = get_ml_sensor_data(body)

        engine = get_ml_engine()

        result = engine.predict_m2(
            esp32_data=data,
            crop_type=crop_type,
            best_time=best_time,
        )

        return jsonify({
            "success": True,
            **result,
        })

    except ValueError as exc:
        return json_error(str(exc), 400)

    except ESP32Error as exc:
        return json_error(str(exc), 503)

    except Exception as exc:
        logger.exception("M2 irrigation prediction failed")
        return json_error(str(exc), 500)


# ------------------------------------------------------------
# M3 - FUTURE SOIL MOISTURE
# ------------------------------------------------------------

@app.route("/api/ml/soil-moisture", methods=["POST"])
def api_ml_soil_moisture():
    """
    Use request readings when supplied and cached telemetry otherwise.
    Never query the synchronous ESP32 server while it may be proxying this
    request to Flask.

    Weather is obtained through weather.py.
    """

    try:
        body = get_json_body()
        if not has_complete_m3_sensor_reading(body):
            cache_age = get_cached_esp32_data_age_seconds()
            if cache_age is None or cache_age > M3_SENSOR_CACHE_MAX_AGE_SECONDS:
                return jsonify({
                    "success": False,
                    "error_code": "stale_sensor_data",
                    "error": "Current sensor telemetry is unavailable or stale.",
                }), 503
        data = get_ml_sensor_data(body)

        weather = get_current_weather_safe()
        weather = dict(weather)
        weather.setdefault("rain_now_mm", weather.get("rainfall_mm", weather.get("precipitation_mm", 0.0)))
        weather.setdefault("rain_sum_6h_mm", get_m3_rain_sum_6h(
            region=str(data.get("site") or data.get("location") or "Chikkamagaluru"),
            current_datetime=datetime.now(ZoneInfo("Asia/Kolkata")),
        ))

        engine = get_ml_engine()

        result = engine.predict_m3(
            esp32_data=data,
            weather_data=weather,
        )

        if not result.get("success"):
            status = {
                "insufficient_history": 409,
                "invalid_features": 400,
            }.get(result.get("error_code"), 500)
            return jsonify(result), status

        return jsonify(normalize_m3_result(result))

    except ESP32Error as exc:
        return json_error(str(exc), 503)

    except Exception as exc:
        logger.exception("M3 soil moisture prediction failed")
        return json_error(str(exc), 500)


# ------------------------------------------------------------
# M4 - FUTURE RAINFALL
# ------------------------------------------------------------

@app.route("/api/ml/rainfall", methods=["POST"])
def api_ml_rainfall():
    """
    Dashboard payload:

    {
        "region": "Chikkamagaluru"
    }
    """

    try:
        body = get_json_body()
        region = str(
            body.get("region")
            or "Chikkamagaluru"
        ).strip()

        if not region:
            return json_error(
                "Region cannot be empty.",
                400,
            )

        try:
            region = normalize_weather_region(region)
        except ValueError as exc:
            return json_error(str(exc), 400)

        engine = get_ml_engine()

        result = engine.predict_m4_latest(
            region=region
        )

        if not result.get("success"):
            status = 503 if str(result.get("error", "")).startswith("Weather unavailable:") else 500
            return jsonify(result), status

        predicted = result.get(
            "predicted_next_hour_rainfall_mm",
            result.get("future_rainfall_mm", result.get("predicted_rainfall_mm")),
        )
        if predicted is None:
            return json_error("Rainfall model returned no prediction value.", 502)

        return jsonify({
            **result,
            "success": True,
            "predicted_next_hour_rainfall_mm": predicted,
        })

    except Exception as exc:
        logger.exception("M4 rainfall prediction failed")
        return json_error(str(exc), 500)


# ------------------------------------------------------------
# ALL LIVE ML RESULTS
# ------------------------------------------------------------

@app.route("/api/ml/live", methods=["POST"])
def api_ml_live():
    """
    Runs M1 -> M2 -> M3 -> M4 using the current ESP32 state.

    Payload:

    {
        "location": "Chikkamagaluru",
        "soil_type": "Red Soil",
        "irrigation": "Drip",
        "crop_type": null,
        "best_time": "Any",
        "region": "Chikkamagaluru"
    }
    """

    try:
        body = get_json_body()

        location = str(
            body.get("location")
            or "Chikkamagaluru"
        )

        soil_type = str(
            body.get("soil_type")
            or "Red Soil"
        )

        irrigation = str(
            body.get("irrigation")
            or "Drip"
        )

        crop_type = body.get("crop_type")
        best_time = str(
            body.get("best_time")
            or "Any"
        )

        region = str(
            body.get("region")
            or location
        )

        data = get_ml_sensor_data(body)

        engine = get_ml_engine()

        result = engine.run_live_predictions(
            esp32_data=data,
            location=normalize_m1_location(location),
            soil_type=soil_type,
            irrigation=irrigation,
            crop_type=crop_type,
            best_time=best_time,
            region=normalize_m1_location(location),
        )

        return jsonify({
            "success": True,
            **result,
        })

    except ESP32Error as exc:
        return json_error(str(exc), 503)

    except Exception as exc:
        logger.exception("Live ML prediction failed")
        return json_error(str(exc), 500)


# ------------------------------------------------------------
# MODEL STATUS
# ------------------------------------------------------------

@app.route("/api/ml/status", methods=["GET"])
def api_ml_status():
    try:
        engine = get_ml_engine()

        return jsonify({
            "success": True,
            "models": engine.model_status(),
        })

    except Exception as exc:
        logger.exception("Could not read ML status")
        return json_error(str(exc), 500)


# ------------------------------------------------------------
# WIFI SETUP / PROVISIONING
# ------------------------------------------------------------

def esp32_direct_request(
    method: str,
    ip: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
):
    """
    Direct proxy used for Wi-Fi setup.

    During provisioning the ESP32 normally lives at:
        192.168.4.1

    Once connected, the returned IP can be stored in the
    ESP32Manager for normal dashboard communication.
    """

    url = f"http://{ip}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            timeout=ESP32_TIMEOUT,
        )
    except requests.RequestException as first_exc:
        # If a stale hostname/IP is supplied, try LAN discovery once.
        discovered = ESP32.discover()
        discovered_ip = discovered.get("ip") if discovered else None
        if discovered_ip and str(discovered_ip) != str(ip):
            ip = str(discovered_ip)
            url = f"http://{ip}{endpoint}"
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    json=payload,
                    timeout=ESP32_TIMEOUT,
                )
            except requests.RequestException as retry_exc:
                raise ESP32ConnectionError(
                    f"Could not reach ESP32 after discovery at {url}: {retry_exc}"
                ) from retry_exc
        else:
            raise ESP32ConnectionError(
                f"Could not reach ESP32 at {url}: {first_exc}"
            ) from first_exc

    if not response.ok:
        raise ESP32APIError(
            f"ESP32 Wi-Fi API returned HTTP "
            f"{response.status_code}: {response.text[:500]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ESP32APIError(
            f"ESP32 returned invalid JSON: {response.text[:500]}"
        ) from exc


@app.route("/api/wifi/scan", methods=["GET"])
def api_wifi_scan():
    """
    Proxies:
        GET /api/wifi/scan

    to the ESP32.

    If ESP32_IP is not configured, use the setup AP IP.
    """

    try:
        ip = esp32_manager.ip or SETUP_AP_IP

        result = esp32_direct_request(
            method="GET",
            ip=ip,
            endpoint="/api/wifi/scan",
        )

        if isinstance(result, list):
            result = {"networks": result}

        if "networks" not in result:
            result["networks"] = []

        return jsonify(result)

    except ESP32Error as exc:
        return json_error(str(exc), 503)

    except Exception as exc:
        logger.exception("Wi-Fi scan failed")
        return json_error(str(exc), 500)


@app.route("/api/wifi/connect", methods=["POST"])
def api_wifi_connect():
    """
    Proxies Wi-Fi credentials to the ESP32.

    IMPORTANT:
    - Password is received only in this request.
    - Backend does not save it.
    - config.py does not contain it.
    - ESP32/WiFiManager owns credential storage.

    Expected request:

    {
        "ssid": "MyWiFi",
        "password": "secret"
    }

    Expected ESP32 response:

    {
        "success": true,
        "ssid": "MyWiFi",
        "ip": "192.168.1.25",
        "security": "WPA2"
    }
    """

    try:
        body = get_json_body()

        ssid = str(require_value(body, "ssid"))
        password = str(body.get("password") or "")

        current_ip = esp32_manager.ip or SETUP_AP_IP

        result = esp32_direct_request(
            method="POST",
            ip=current_ip,
            endpoint="/api/wifi/connect",
            payload={
                "ssid": ssid,
                "password": password,
            },
        )

        # Never log password.
        logger.info(
            "Wi-Fi connection requested for SSID '%s'.",
            ssid,
        )

        new_ip = result.get("ip")

        if result.get("success") and new_ip:
            try:
                esp32_manager.set_ip(str(new_ip))
            except Exception as exc:
                logger.warning(
                    "Wi-Fi connected but backend could not "
                    "switch to new ESP32 IP: %s",
                    exc,
                )

        # Remove accidental password echo from firmware response.
        safe_result = dict(result)
        safe_result.pop("password", None)

        return jsonify(safe_result)

    except ValueError as exc:
        return json_error(str(exc), 400)

    except ESP32Error as exc:
        return json_error(str(exc), 503)

    except Exception as exc:
        logger.exception("Wi-Fi connection request failed")
        return json_error(str(exc), 500)


# ------------------------------------------------------------
# OPTIONS
# ------------------------------------------------------------

@app.route(
    "/api/<path:_path>",
    methods=["OPTIONS"],
)
def api_options(_path):
    return ("", 204)


# ------------------------------------------------------------
# GLOBAL ERROR HANDLERS
# ------------------------------------------------------------

@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found.",
    }), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({
        "success": False,
        "error": "HTTP method not allowed.",
    }), 405


@app.errorhandler(Exception)
def unhandled_error(error):
    logger.exception("Unhandled backend error")

    return jsonify({
        "success": False,
        "error": str(error),
    }), 500


# ------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("KRUSHI MITHRA BACKEND")
    print("=" * 70)
    print(f"System : {SYSTEM_NAME}")
    print(f"Version: {SYSTEM_VERSION}")
    print(f"ESP32  : {esp32_manager.ip or 'AUTO-DISCOVERY'}")
    print("Discovery: UDP 4210 + mDNS krushi-mithra.local")
    print()
    print("Dashboard : http://127.0.0.1:5000/")
    print("Wi-Fi UI  : http://127.0.0.1:5000/wifi-setup")
    print()
    print("API:")
    print("  GET  /api/health")
    print("  GET  /api/backend/status")
    print("  GET  /api/device/discover")
    print("  GET  /api/data")
    print("  GET  /api/status")
    print("  GET  /api/config")
    print("  POST /api/system")
    print("  POST /api/mode")
    print("  POST /api/pump")
    print("  POST /api/config")
    print("  POST /api/ml/crop")
    print("  POST /api/ml/irrigation")
    print("  POST /api/ml/soil-moisture")
    print("  POST /api/ml/rainfall")
    print("  POST /api/ml/live")
    print("  GET  /api/ml/status")
    print("  GET  /api/wifi/scan")
    print("  POST /api/wifi/connect")
    print("=" * 70)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
