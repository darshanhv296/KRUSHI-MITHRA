import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_FILE = Path("karnataka_balanced_crops_3000.csv")
WEATHER_CACHE = Path("karnataka_base_weather_cache.csv")

START_YEAR = 2014
END_YEAR = 2025
SAMPLES_PER_CROP = 110  # ~110 samples per crop -> ~3,080 perfectly balanced rows

# ============================================================
# 1. 31 KARNATAKA DISTRICTS WITH STANDARDIZED CROPS
# (Synonyms merged: Red Gram/Pigeon Pea -> Red Gram; Pepper -> Black Pepper)
# ============================================================
DISTRICT_PROFILES = [
    # North Eastern Transition & Dry Zones
    {
        "Location": "Bidar",
        "Lat": 17.9104,
        "Lon": 77.5199,
        "Soil_Type": "Red Laterite",
        "Crops": ["Jowar", "Red Gram", "Bajra", "Sugarcane", "Soybean"],
    },
    {
        "Location": "Kalaburagi",
        "Lat": 17.3297,
        "Lon": 76.8343,
        "Soil_Type": "Deep Black Clay",
        "Crops": ["Red Gram", "Jowar", "Bajra", "Bengal Gram", "Cotton"],
    },
    {
        "Location": "Yadgir",
        "Lat": 16.7700,
        "Lon": 77.1300,
        "Soil_Type": "Medium Black Clay",
        "Crops": ["Red Gram", "Jowar", "Cotton", "Sunflower", "Bajra"],
    },
    {
        "Location": "Raichur",
        "Lat": 16.2076,
        "Lon": 77.3463,
        "Soil_Type": "Deep Black Clay",
        "Crops": ["Paddy", "Cotton", "Jowar", "Bajra", "Groundnut"],
    },
    # Northern Dry Zone
    {
        "Location": "Bagalkot",
        "Lat": 16.1817,
        "Lon": 75.6958,
        "Soil_Type": "Medium Black Clay",
        "Crops": ["Sugarcane", "Maize", "Jowar", "Bengal Gram", "Sunflower"],
    },
    {
        "Location": "Vijayapura",
        "Lat": 16.8302,
        "Lon": 75.7100,
        "Soil_Type": "Deep Black Clay",
        "Crops": ["Jowar", "Bajra", "Grapes", "Bengal Gram", "Maize"],
    },
    {
        "Location": "Gadag",
        "Lat": 15.4319,
        "Lon": 75.6355,
        "Soil_Type": "Deep Black Clay",
        "Crops": ["Bengal Gram", "Jowar", "Cotton", "Groundnut", "Onion"],
    },
    {
        "Location": "Koppal",
        "Lat": 15.3456,
        "Lon": 76.1554,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Paddy", "Maize", "Jowar", "Bajra", "Pomegranate"],
    },
    {
        "Location": "Ballari",
        "Lat": 15.1394,
        "Lon": 76.9214,
        "Soil_Type": "Deep Black Clay",
        "Crops": ["Paddy", "Cotton", "Chilli", "Maize", "Jowar"],
    },
    {
        "Location": "Vijayanagara",
        "Lat": 15.2700,
        "Lon": 76.3900,
        "Soil_Type": "Medium Black Clay",
        "Crops": ["Paddy", "Sugarcane", "Maize", "Cotton", "Banana"],
    },
    # Northern Transition Zone
    {
        "Location": "Belagavi",
        "Lat": 15.8497,
        "Lon": 74.4977,
        "Soil_Type": "Medium Black Clay",
        "Crops": ["Sugarcane", "Paddy", "Soybean", "Maize", "Cotton"],
    },
    {
        "Location": "Dharwad",
        "Lat": 15.4589,
        "Lon": 75.0078,
        "Soil_Type": "Medium Black Clay",
        "Crops": ["Cotton", "Wheat", "Soybean", "Jowar", "Chilli"],
    },
    {
        "Location": "Haveri",
        "Lat": 14.7954,
        "Lon": 75.3995,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Chilli", "Maize", "Cotton", "Jowar", "Groundnut"],
    },
    # Central Dry Zone
    {
        "Location": "Chitradurga",
        "Lat": 14.2251,
        "Lon": 76.4022,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Groundnut", "Ragi", "Maize", "Arecanut", "Pomegranate"],
    },
    {
        "Location": "Davanagere",
        "Lat": 14.4644,
        "Lon": 75.9218,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Maize", "Paddy", "Arecanut", "Sugarcane", "Ragi"],
    },
    {
        "Location": "Tumakuru",
        "Lat": 13.3379,
        "Lon": 77.1010,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Coconut", "Arecanut", "Ragi", "Groundnut", "Horse Gram"],
    },
    # Eastern Dry Zone
    {
        "Location": "Kolar",
        "Lat": 13.1367,
        "Lon": 78.1291,
        "Soil_Type": "Red Loam",
        "Crops": ["Tomato", "Ragi", "Mulberry", "Mango", "Groundnut"],
    },
    {
        "Location": "Chikkaballapura",
        "Lat": 13.4325,
        "Lon": 77.7275,
        "Soil_Type": "Red Loam",
        "Crops": ["Mulberry", "Ragi", "Maize", "Grapes", "Tomato"],
    },
    {
        "Location": "Bengaluru Urban",
        "Lat": 12.9716,
        "Lon": 77.5946,
        "Soil_Type": "Red Loam",
        "Crops": ["Ragi", "Vegetables", "Mulberry", "Maize", "Tomato"],
    },
    {
        "Location": "Bengaluru Rural",
        "Lat": 13.2257,
        "Lon": 77.5750,
        "Soil_Type": "Red Loam",
        "Crops": ["Ragi", "Mulberry", "Grapes", "Vegetables", "Maize"],
    },
    {
        "Location": "Ramanagara",
        "Lat": 12.7209,
        "Lon": 77.2799,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Mulberry", "Ragi", "Mango", "Coconut", "Red Gram"],
    },
    # Southern Dry Zone
    {
        "Location": "Mandya",
        "Lat": 12.5218,
        "Lon": 76.8951,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Sugarcane", "Paddy", "Ragi", "Horse Gram", "Banana"],
    },
    {
        "Location": "Mysuru",
        "Lat": 12.2958,
        "Lon": 76.6394,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Paddy", "Ragi", "Tobacco", "Sugarcane", "Cotton"],
    },
    {
        "Location": "Chamarajanagar",
        "Lat": 11.9261,
        "Lon": 76.9437,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Turmeric", "Banana", "Sugarcane", "Ragi", "Maize"],
    },
    # Southern Transition Zone
    {
        "Location": "Hassan",
        "Lat": 13.0033,
        "Lon": 76.1004,
        "Soil_Type": "Red Sandy Loam",
        "Crops": ["Coffee", "Potato", "Ragi", "Paddy", "Black Pepper"],
    },
    {
        "Location": "Shivamogga",
        "Lat": 13.9299,
        "Lon": 75.5681,
        "Soil_Type": "Laterite",
        "Crops": ["Arecanut", "Paddy", "Maize", "Ginger", "Black Pepper"],
    },
    # Hilly Zone (Malnad)
    {
        "Location": "Chikkamagaluru",
        "Lat": 13.3153,
        "Lon": 75.7754,
        "Soil_Type": "Red Laterite",
        "Crops": ["Coffee", "Cardamom", "Black Pepper", "Arecanut", "Tea"],
    },
    {
        "Location": "Kodagu",
        "Lat": 12.3375,
        "Lon": 75.8069,
        "Soil_Type": "Red Clay Loam",
        "Crops": ["Coffee", "Black Pepper", "Cardamom", "Orange", "Rubber"],
    },
    # Coastal Zone
    {
        "Location": "Dakshina Kannada",
        "Lat": 12.8703,
        "Lon": 74.8806,
        "Soil_Type": "Coastal Alluvial & Laterite",
        "Crops": ["Arecanut", "Coconut", "Paddy", "Cashew", "Rubber"],
    },
    {
        "Location": "Udupi",
        "Lat": 13.3409,
        "Lon": 74.7421,
        "Soil_Type": "Coastal Alluvial & Laterite",
        "Crops": ["Paddy", "Coconut", "Arecanut", "Cashew", "Black Pepper"],
    },
    {
        "Location": "Uttara Kannada",
        "Lat": 14.7950,
        "Lon": 74.6860,
        "Soil_Type": "Laterite",
        "Crops": ["Arecanut", "Paddy", "Ginger", "Cashew", "Sugarcane"],
    },
]

# ============================================================
# 2. OPEN-METEO WEATHER FETCH (Historical Archive)
# ============================================================
session = requests.Session()
session.headers.update({"User-Agent": "Karnataka-Agri-Research/2.0"})


def fetch_weather(lat, lon, year, max_retries=3):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": [
            "precipitation_sum",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "soil_moisture_0_to_7cm_mean",
        ],
        "timezone": "auto",
    }
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=40)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json().get("daily", {})

            rain = [v for v in d.get("precipitation_sum", []) if v is not None]
            temp = [v for v in d.get("temperature_2m_mean", []) if v is not None]
            hum = [
                v
                for v in d.get("relative_humidity_2m_mean", [])
                if v is not None
            ]
            sm = [
                v
                for v in d.get("soil_moisture_0_to_7cm_mean", [])
                if v is not None
            ]

            return {
                "Annual_Rainfall_mm": round(float(np.sum(rain)), 2),
                "Temperature": round(float(np.mean(temp)), 2),
                "Humidity": round(float(np.mean(hum)), 2),
                "Soil_Moisture": (
                    round(float(np.mean(sm)) * 100, 2) if sm else 26.5
                ),
            }
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2)


# ============================================================
# 3. BASELINE WEATHER CACHE
# ============================================================
if WEATHER_CACHE.exists():
    cache_df = pd.read_csv(WEATHER_CACHE)
else:
    cache_df = pd.DataFrame(
        columns=[
            "Location",
            "Year",
            "Annual_Rainfall_mm",
            "Temperature",
            "Humidity",
            "Soil_Moisture",
        ]
    )

cached_keys = set(zip(cache_df["Location"], cache_df["Year"].astype(int)))
weather_rows = cache_df.to_dict("records")

print("Checking Open-Meteo baseline weather data (2014-2025)...")
for d in DISTRICT_PROFILES:
    loc = d["Location"]
    for y in range(START_YEAR, END_YEAR + 1):
        if (loc, y) in cached_keys:
            continue
        stats = fetch_weather(d["Lat"], d["Lon"], y)
        if stats:
            weather_rows.append({"Location": loc, "Year": y, **stats})
            pd.DataFrame(weather_rows).to_csv(WEATHER_CACHE, index=False)
        time.sleep(0.12)

base_weather_df = pd.DataFrame(weather_rows)

# ============================================================
# 4. PREPARE BASE OBSERVATIONS
# ============================================================
meta_df = pd.DataFrame(DISTRICT_PROFILES)[["Location", "Soil_Type", "Crops"]]
base_merged = base_weather_df.merge(meta_df, on="Location", how="inner")
base_exploded = base_merged.explode("Crops").rename(columns={"Crops": "Crop"})

# ============================================================
# 5. STRATIFIED BALANCING (~110 SAMPLES PER CROP)
# ============================================================
print(
    f"\nBalancing dataset to approximately {SAMPLES_PER_CROP} samples per crop..."
)

unique_crops = sorted(base_exploded["Crop"].unique())
balanced_rows = []
rng = np.random.default_rng(seed=42)

for crop in unique_crops:
    crop_pool = base_exploded[base_exploded["Crop"] == crop].reset_index(
        drop=True
    )
    n_existing = len(crop_pool)

    # Oversample / sample with replacement across available real agro-climate years
    selected_indices = rng.choice(n_existing, size=SAMPLES_PER_CROP, replace=True)

    for idx in selected_indices:
        row = crop_pool.iloc[idx]

        # Natural farm-level microclimate variation (topography, local showers)
        rain_noise = rng.normal(1.0, 0.04)  # ±4%
        temp_noise = rng.normal(0.0, 0.35)  # ±0.35°C
        hum_noise = rng.normal(0.0, 1.2)  # ±1.2%
        sm_noise = rng.normal(1.0, 0.035)  # ±3.5%

        balanced_rows.append({
            "Location": row["Location"],
            "Year": int(row["Year"]),
            "Soil_Type": row["Soil_Type"],
            "Annual_Rainfall_mm": round(
                max(200.0, row["Annual_Rainfall_mm"] * rain_noise), 2
            ),
            "Temperature": round(
                np.clip(row["Temperature"] + temp_noise, 12.0, 45.0), 2
            ),
            "Humidity": round(
                np.clip(row["Humidity"] + hum_noise, 20.0, 99.0), 2
            ),
            "Soil_Moisture": round(
                np.clip(row["Soil_Moisture"] * sm_noise, 5.0, 85.0), 2
            ),
            "Crop": crop,
        })

final_df = pd.DataFrame(balanced_rows)

# Shuffle rows so classes are mixed uniformly
final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

final_columns = [
    "Location",
    "Year",
    "Soil_Type",
    "Annual_Rainfall_mm",
    "Temperature",
    "Humidity",
    "Soil_Moisture",
    "Crop",
]
final_df = final_df[final_columns]
final_df.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# 6. VERIFICATION OF CLASS BALANCE
# ============================================================
print("\n" + "=" * 65)
print("CLASS DISTRIBUTION PER CROP (ALL BALANCED):")
print("=" * 65)
print(final_df["Crop"].value_counts().to_string())

print("\n" + "=" * 65)
print("DATASET DETAILS:")
print("=" * 65)
print(f"Total Records  : {len(final_df):,}")
print(f"Total Classes  : {final_df['Crop'].nunique()}")
print(f"Target Schema  : {list(final_df.columns)}")
print(f"Saved To       : {OUTPUT_FILE.resolve()}")
print("=" * 65)