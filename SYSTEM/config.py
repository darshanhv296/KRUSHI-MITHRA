"""
Krushi Mithra
Smart Farming & Intelligent Irrigation System

Runtime configuration
"""

from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

SYSTEM_NAME = "Krushi Mithra"
SYSTEM_VERSION = "1.0.0"
TIMEZONE = "Asia/Kolkata"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "system.log"

MODEL_DIR_CANDIDATES = [
    PROJECT_ROOT / "models" / "TRAINED_MODELS",
    PROJECT_ROOT / "MODELS" / "TRAINED_MODELS",
    PROJECT_ROOT / "DATASET" / "TRAINED_MODELS",
]
MODEL_DIR = next((p for p in MODEL_DIR_CANDIDATES if p.exists()), MODEL_DIR_CANDIDATES[0])

M4_WEATHER_HISTORY_FILE = DATA_DIR / "M4_weather_history.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ESP32 CONNECTION
# ============================================================

ESP32_HOSTNAME = "krushi-mithra.local"
ESP32_IP = "10.46.117.138"

ESP32_TIMEOUT = 5.0
BACKEND_HEARTBEAT_INTERVAL = 10.0

# ============================================================
# ESP32 NETWORK DISCOVERY
# ============================================================

ESP32_DISCOVERY_PORT = 4210
ESP32_DISCOVERY_MESSAGE = "KRUSHI_MITHRA_DISCOVER"
ESP32_DISCOVERY_TIMEOUT = 2.0

# ============================================================
# ESP32 / REAL-TIME POLLING
# ============================================================

ESP32_POLL_INTERVAL = 2
ESP32_STATUS_POLL_INTERVAL = 10


# ============================================================
# ML RUNTIME INTERVALS
# ============================================================

M2_PREDICTION_INTERVAL = 60
M3_PREDICTION_INTERVAL = 60
M4_PREDICTION_INTERVAL = 60


# ============================================================
# ML / WEATHER DATA DATE RANGE
# ============================================================

TRAINING_END_DATE = "2026-09-30"
RUNTIME_START_DATE = "2026-10-01"


# ============================================================
# WEATHER RUNTIME
# ============================================================

WEATHER_CURRENT_INTERVAL = 15 * 60


# ============================================================
# M4 WEATHER HISTORY SYNCHRONIZATION
# ============================================================

M4_SYNC_HOUR_1 = 8
M4_SYNC_HOUR_2 = 20


# ============================================================
# WEATHER LOCATION
# ============================================================

WEATHER_LATITUDE = 13.3223
WEATHER_LONGITUDE = 75.7740
WEATHER_TIMEZONE = "Asia/Kolkata"


# ============================================================
# OPEN-METEO API
# ============================================================

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# M1 — CROP RECOMMENDATION
# ============================================================

M1_LOCATION = None
M1_SOIL_TYPE = None
M1_IRRIGATION = None


# ============================================================
# IRRIGATION CONTROL CONFIGURATION
# ============================================================

SOIL_TRIGGER = 40.0
CRITICAL_SOIL = 25.0
RAIN_PROTECTION = True


# ============================================================
# SYSTEM DEFAULTS
# ============================================================

SYSTEM_ENABLED_DEFAULT = False
MODE_DEFAULT = "AUTO"
PUMP_DEFAULT = False


# ============================================================
# VALIDATION
# ============================================================

if MODE_DEFAULT not in ("AUTO", "MANUAL"):
    raise ValueError("MODE_DEFAULT must be either 'AUTO' or 'MANUAL'.")


if not 0 <= SOIL_TRIGGER <= 100:
    raise ValueError(
        "SOIL_TRIGGER must be between 0 and 100."
    )


if not 0 <= CRITICAL_SOIL <= 100:
    raise ValueError(
        "CRITICAL_SOIL must be between 0 and 100."
    )


if CRITICAL_SOIL > SOIL_TRIGGER:
    raise ValueError(
        "CRITICAL_SOIL cannot be greater than SOIL_TRIGGER."
    )


# ============================================================
# END OF CONFIGURATION
# ============================================================