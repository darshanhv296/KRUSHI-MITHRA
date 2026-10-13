"""
Open-Meteo 3-Year Hourly Weather & Rain Dataset for all 31 Karnataka Districts.

Outputs:
  karnataka_3years_hourly_all_districts.csv -> Single unified CSV with all 31 districts across 3 full years (2023-2025).

Install:
  pip install openmeteo-requests requests-cache retry-requests pandas numpy
"""

from pathlib import Path
import time
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
START_YEAR = 2023
END_YEAR = 2025
TIMEZONE = "Asia/Kolkata"

try:
    OUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUT_DIR = Path.cwd()

OUTPUT_FILE = OUT_DIR / "karnataka_3years_hourly_all_districts.csv"
CACHE_DIR = OUT_DIR / "raw_cache"

DISTRICTS = {
    "Bagalkot": (16.1817, 75.6958),
    "Ballari": (15.1394, 76.9214),
    "Belagavi": (15.8497, 74.4977),
    "Bengaluru Rural": (13.2957, 77.5376),
    "Bengaluru Urban": (12.9716, 77.5946),
    "Bidar": (17.9104, 77.5199),
    "Chamarajanagar": (11.9261, 76.9437),
    "Chikkaballapura": (13.4355, 77.7315),
    "Chikkamagaluru": (13.3223, 75.7740),
    "Chitradurga": (14.2251, 76.3980),
    "Dakshina Kannada": (12.9141, 74.8560),
    "Davanagere": (14.4644, 75.9218),
    "Dharwad": (15.4589, 75.0078),
    "Gadag": (15.4298, 75.6297),
    "Hassan": (13.0033, 76.1004),
    "Haveri": (14.7951, 75.4044),
    "Kalaburagi": (17.3297, 76.8343),
    "Kodagu": (12.4244, 75.7382),
    "Kolar": (13.1360, 78.1294),
    "Koppal": (15.3459, 76.1548),
    "Mandya": (12.5218, 76.8951),
    "Mysuru": (12.2958, 76.6394),
    "Raichur": (16.2120, 77.3439),
    "Ramanagara": (12.7159, 77.2815),
    "Shivamogga": (13.9299, 75.5681),
    "Tumakuru": (13.3392, 77.1130),
    "Udupi": (13.3409, 74.7421),
    "Uttara Kannada": (14.7937, 74.6869),
    "Vijayanagara": (15.2689, 76.3909),
    "Vijayapura": (16.8302, 75.7100),
    "Yadgir": (16.7625, 77.1380),
}

HOURLY_VARS = [
    "rain",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
]
COLS = [
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "pressure_msl_hpa",
    "cloud_cover_percent",
    "wind_speed_kmh",
]

_client = None


def _get_client():
    global _client
    if _client is None:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        session = retry(
            requests_cache.CachedSession(".cache", expire_after=-1),
            retries=5,
            backoff_factor=0.5,
        )
        _client = openmeteo_requests.Client(session=session)
    return _client


def fetch_district_year(name, lat, lon, year):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{name.replace(' ', '_')}_{year}.csv"

    # If already cached, load and guarantee 'district' is present
    if cache_file.exists():
        df_cached = pd.read_csv(cache_file, parse_dates=["timestamp"])
        df_cached["district"] = name
        return df_cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": HOURLY_VARS,
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "cell_selection": "land",
    }

    for attempt in range(4):
        try:
            r = _get_client().weather_api(
                "https://archive-api.open-meteo.com/v1/archive", params=params
            )[0]
            break
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(
                f"  [{name} {year}] {e} -> waiting {wait}s (attempt {attempt + 1}/4)"
            )
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed {name} {year}.")

    h = r.Hourly()
    ts = pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left",
    )
    ts = ts.tz_convert(TIMEZONE).tz_localize(None)

    df = pd.DataFrame({"district": name, "timestamp": ts})
    for i, c in enumerate(COLS):
        df[c] = h.Variables(i).ValuesAsNumpy().round(2)

    df.to_csv(cache_file, index=False)
    time.sleep(0.2)
    return df


def main():
    all_frames = []
    total_districts = len(DISTRICTS)
    print(
        f"Processing {START_YEAR}-{END_YEAR} hourly weather for all {total_districts} Karnataka districts...\n"
    )

    for i, (name, (lat, lon)) in enumerate(DISTRICTS.items(), 1):
        print(f"[{i:02d}/{total_districts}] Processing {name}...")
        for y in range(START_YEAR, END_YEAR + 1):
            df_year = fetch_district_year(name, lat, lon, y)
            df_year["district"] = name  # Guarantees district column exists
            all_frames.append(df_year)

    final_df = pd.concat(all_frames, ignore_index=True)

    # Time components
    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])
    final_df["date"] = final_df["timestamp"].dt.strftime("%Y-%m-%d")
    final_df["time"] = final_df["timestamp"].dt.strftime("%H:%M:%S")
    final_df["year"] = final_df["timestamp"].dt.year
    final_df["month"] = final_df["timestamp"].dt.month
    final_df["hour"] = final_df["timestamp"].dt.hour
    final_df["day_of_year"] = final_df["timestamp"].dt.dayofyear

    columns_order = [
        "district",
        "timestamp",
        "date",
        "time",
        "year",
        "month",
        "day_of_year",
        "hour",
        "rainfall_mm",
        "temperature_c",
        "humidity_percent",
        "pressure_msl_hpa",
        "cloud_cover_percent",
        "wind_speed_kmh",
    ]
    final_df = final_df[columns_order]
    final_df = final_df.sort_values(["district", "timestamp"]).reset_index(
        drop=True
    )

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("3-YEAR DATASET CREATED SUCCESSFULLY")
    print("=" * 65)
    print(f"Total Rows Generated : {len(final_df):,}")
    print(f"Districts Covered    : {final_df['district'].nunique()} districts")
    print(f"Time Horizon         : {START_YEAR} to {END_YEAR} (3 full years)")
    print(
        f"Hours Per District   : {len(final_df) // final_df['district'].nunique():,}"
    )
    print(f"Saved To             : {OUTPUT_FILE.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()