import time
from pathlib import Path
import pandas as pd
import requests

OUTPUT_FILE = Path("real_soil_moisture_prediction_dataset.csv")
CACHE_FILE = Path("real_soil_moisture_cache.csv")

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# Real coordinates across representative agricultural regions
REGIONS = [
    {"name": "Central_Dry_Tiptur", "lat": 13.2558, "lon": 76.4785},
    {"name": "Northern_Clay_Kalaburagi", "lat": 17.3297, "lon": 76.8343},
    {"name": "Hilly_Malnad_Chikkamagaluru", "lat": 13.3153, "lon": 75.7754},
    {"name": "Coastal_Udupi", "lat": 13.3409, "lon": 74.7421},
]

session = requests.Session()
session.headers.update({"User-Agent": "RealSoilMoistureDataset/1.0"})


def fetch_real_hourly_data(lat, lon, start_date, end_date, max_retries=3):
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
        except Exception:
            if attempt == max_retries - 1:
                return {}
            time.sleep(3)


# Load or fetch raw observations
if CACHE_FILE.exists():
    raw_df = pd.read_csv(CACHE_FILE)
    cached_regions = set(raw_df["region"].unique())
    records = raw_df.to_dict("records")
else:
    cached_regions = set()
    records = []

for reg in REGIONS:
    name = reg["name"]
    if name in cached_regions:
        continue

    print(f"Fetching real observational data for {name}...")
    h = fetch_real_hourly_data(reg["lat"], reg["lon"], START_DATE, END_DATE)

    times = h.get("time", [])
    temps = h.get("temperature_2m", [])
    hums = h.get("relative_humidity_2m", [])
    st = h.get("soil_temperature_0_to_7cm", [])
    sms = h.get("soil_moisture_0_to_7cm", [])
    precips = h.get("precipitation", [])

    for i in range(len(times)):
        if (
            times[i] is not None
            and temps[i] is not None
            and hums[i] is not None
            and st[i] is not None
            and sms[i] is not None
        ):
            records.append({
                "region": name,
                "timestamp_str": times[i],
                "air_temperature": round(float(temps[i]), 2),
                "air_humidity": round(float(hums[i]), 2),
                "soil_temperature": round(float(st[i]), 2),
                # Real volumetric soil moisture m³/m³ converted to percentage (0 - 100%)
                "soil_moisture": round(float(sms[i]) * 100.0, 2),
                "rainfall_mm": round(
                    float(precips[i]) if precips[i] is not None else 0.0, 2
                ),
            })

    pd.DataFrame(records).to_csv(CACHE_FILE, index=False)
    time.sleep(0.5)

df = pd.DataFrame(records)

# ============================================================
# EXTRACT THE EXACT SCHEMA FROM YOUR IMAGE
# ============================================================
# Parse ISO timestamp (e.g. 2024-05-10T14:00)
dt_series = pd.to_datetime(df["timestamp_str"])

df["time"] = dt_series.dt.strftime("%H:%M:%S")
df["date"] = dt_series.dt.strftime("%Y-%m-%d")
df["hour"] = dt_series.dt.hour
df["day_of_year"] = dt_series.dt.dayofyear
df["month"] = dt_series.dt.month
df["air_tempe"] = df["air_temperature"]
df["air_humidi"] = df["air_humidity"]
df["soil_tempe"] = df["soil_temperature"]
df["soil_moist"] = df["soil_moisture"]
df["rainfall_m"] = df["rainfall_mm"]

# Exact columns as displayed in your image header
final_columns = [
    "time",
    "date",
    "hour",
    "day_of_year",
    "month",
    "air_tempe",
    "air_humidi",
    "soil_tempe",
    "soil_moist",
    "rainfall_m",
]

final_df = df[final_columns].dropna().reset_index(drop=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("REAL-WORLD SOIL MOISTURE DATASET CREATED")
print("=" * 65)
print(f"Total Real Rows : {len(final_df):,}")
print(f"Columns         : {list(final_df.columns)}")
print(f"Soil Moist Range: {final_df['soil_moist'].min()}% to {final_df['soil_moist'].max()}%")
print(f"Saved To        : {OUTPUT_FILE.resolve()}")
print("=" * 65)
print("\nFirst 5 Records:")
print(final_df.head())