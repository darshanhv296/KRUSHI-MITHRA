from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIGURATION - TUMAKURU / TIPTUR DISTRICT
# ============================================================
OUTPUT_FILE = Path("tumakuru_irrigation_clean_hysteresis.csv")
CACHE_FILE = Path("tumakuru_hourly_raw_cache.csv")

LATITUDE = 13.3379
LONGITUDE = 77.1010
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# Hysteresis boundaries (percentages)
MOISTURE_ON_THRESH = 25.0  # Turn ON when soil drops below this
MOISTURE_OFF_THRESH = 30.0  # Turn OFF when soil recovers above this

# ============================================================
# 1. FETCH TUMAKURU HOURLY DATA FROM OPEN-METEO ARCHIVE
# ============================================================
session = requests.Session()
session.headers.update({"User-Agent": "Tumakuru-Irrigation-Pipeline/1.0"})


def fetch_tumakuru_hourly(lat, lon, start_date, end_date, max_retries=3):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "soil_temperature_0_to_7cm",
            "soil_moisture_0_to_7cm",
            "precipitation",
        ],
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("hourly", {})
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to fetch data: {e}")
            time.sleep(3)


if CACHE_FILE.exists():
    print(f"Loading cached observations from {CACHE_FILE}...")
    df_raw = pd.read_csv(CACHE_FILE)
else:
    print(f"Retrieving raw hourly data for Tumakuru ({START_DATE} to {END_DATE})...")
    raw = fetch_tumakuru_hourly(LATITUDE, LONGITUDE, START_DATE, END_DATE)

    df_raw = pd.DataFrame({
        "timestamp": raw.get("time", []),
        "air_temperature_c": raw.get("temperature_2m", []),
        "air_humidity_pct": raw.get("relative_humidity_2m", []),
        "soil_temperature_c": raw.get("soil_temperature_0_to_7cm", []),
        "soil_moisture_raw": raw.get("soil_moisture_0_to_7cm", []),
        "rainfall_mm": raw.get("precipitation", []),
    })
    df_raw.to_csv(CACHE_FILE, index=False)

# ============================================================
# 2. FEATURE CLEANING & RAINFALL ACCUMULATION
# ============================================================
df = df_raw.dropna().copy().reset_index(drop=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Convert volumetric moisture (m3/m3) to percentage
df["soil_moisture_pct"] = (df["soil_moisture_raw"] * 100.0).round(2)
df["air_temperature_c"] = df["air_temperature_c"].round(2)
df["air_humidity_pct"] = df["air_humidity_pct"].round(2)
df["soil_temperature_c"] = df["soil_temperature_c"].round(2)
df["rainfall_mm"] = df["rainfall_mm"].round(2)

# Compute 6-hour cumulative rain to explain sudden soil moisture jumps
df["rain_last_6h_mm"] = (
    df["rainfall_mm"].rolling(window=6, min_periods=1).sum().round(2)
)

# Extract temporal components
df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear

# ============================================================
# 3. COMPUTE HYSTERESIS PUMP STATE (NO RAPID CHATTERING)
# ============================================================
pump_state = 0
pump_series = []

for idx, row in df.iterrows():
    sm = row["soil_moisture_pct"]
    rain = row["rainfall_mm"]

    # Invalidation: immediate shutoff if it is currently raining
    if rain >= 1.0:
        pump_state = 0
    elif sm < MOISTURE_ON_THRESH:
        pump_state = 1
    elif sm >= MOISTURE_OFF_THRESH:
        pump_state = 0
    # In the 25.0% - 30.0% band, pump_state remains unchanged

    pump_series.append(pump_state)

df["water_pump_status"] = pump_series

# ============================================================
# 4. FINAL CLEANUP & EXPORT
# ============================================================
final_cols = [
    "timestamp",
    "date",
    "time",
    "hour",
    "day_of_year",
    "month",
    "air_temperature_c",
    "air_humidity_pct",
    "soil_temperature_c",
    "rainfall_mm",
    "rain_last_6h_mm",
    "soil_moisture_pct",
    "water_pump_status",
]

df = df[final_cols]
df.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# 5. VERIFICATION REPORT
# ============================================================
print("\n" + "=" * 65)
print("TUMAKURU IRRIGATION DATASET WITH HYSTERESIS COMPLETE")
print("=" * 65)
print(f"Total Hourly Records : {len(df):,}")
print(f"Date Span            : {df['date'].min()} to {df['date'].max()}")
print(
    f"Moisture Range       : {df['soil_moisture_pct'].min()}% to {df['soil_moisture_pct'].max()}%"
)
print(
    f"Pump ON  (Status = 1): {(df['water_pump_status'] == 1).sum():,} hours ({(df['water_pump_status'] == 1).mean() * 100:.1f}%)"
)
print(
    f"Pump OFF (Status = 0): {(df['water_pump_status'] == 0).sum():,} hours ({(df['water_pump_status'] == 0).mean() * 100:.1f}%)"
)

# Verify stability inside the transition band
band_df = df[
    (df["soil_moisture_pct"] >= MOISTURE_ON_THRESH)
    & (df["soil_moisture_pct"] < MOISTURE_OFF_THRESH)
]
print(f"\nSamples in Transition Band (25%-30%): {len(band_df):,}")
print(f"  - Stable ON  in band: {(band_df['water_pump_status'] == 1).sum():,}")
print(f"  - Stable OFF in band: {(band_df['water_pump_status'] == 0).sum():,}")
print(f"Saved to: {OUTPUT_FILE.resolve()}")
print("=" * 65)