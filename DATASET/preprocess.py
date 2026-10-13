import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEANED_DIR = os.path.join(BASE_DIR, "CLEANED")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "PREPROCESSED")

M1_OUT = os.path.join(PREPROCESSED_DIR, "M1_Crop_Recommendation")
M2_OUT = os.path.join(PREPROCESSED_DIR, "M2_Context_Aware_Irrigation")
M3_OUT = os.path.join(PREPROCESSED_DIR, "M3_Future_Soil_Moisture")
M4_OUT = os.path.join(PREPROCESSED_DIR, "M4_Rainfall_Forecasting")

os.makedirs(PREPROCESSED_DIR, exist_ok=True)


# ============================================================
# FINAL CLEANED DATASET FILES
# ============================================================

M1_FILE = os.path.join(CLEANED_DIR, "Crop_Recommendation_cleaned.csv")
M2_FILE = os.path.join(CLEANED_DIR, "Smart_Irigation_cleaned.csv")
M3_FILE = os.path.join(CLEANED_DIR, "Soil_Moisture_Prediction_cleaned.csv")
M4_FILE = os.path.join(CLEANED_DIR, "Rain_Prediction_cleaned.csv")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    """Standardize column names."""
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\nFile not found:\n{path}")


def check_required_columns(df, required, module_name):
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"\n{module_name}: Missing required columns:\n"
            + "\n".join(missing)
        )


def check_no_missing(df, module_name):
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print("\nMissing values:")
        print(missing)
        raise ValueError(
            f"\n{module_name}: Missing values found after preprocessing."
        )


def save_split(train_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train saved : {train_path}")
    print(f"Test saved  : {test_path}")


def normalize_datetime(series, column_name, module_name):
    """Convert timestamps safely to pandas datetime."""
    original = series.copy()

    values = (
        series.astype(str)
        .str.strip()
        .replace(
            ["", "nan", "NaN", "None", "NULL", "null"],
            np.nan
        )
    )

    parsed = pd.to_datetime(
        values,
        errors="coerce",
        format="mixed"
    )

    failed = parsed.isna() & values.notna()

    if failed.any():
        bad_values = (
            original[failed]
            .astype(str)
            .unique()
        )

        print(
            f"\n{module_name}: Invalid values found in "
            f"'{column_name}':"
        )

        for value in bad_values[:20]:
            print("  ", value)

        raise ValueError(
            f"\n{module_name}: Could not convert all "
            f"'{column_name}' values to datetime."
        )

    return parsed


# ============================================================
# MODULE 1
# CROP RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("MODULE 1 - CROP RECOMMENDATION")
print("=" * 70)

check_file_exists(M1_FILE)

m1 = pd.read_csv(M1_FILE)
m1 = clean_column_names(m1)

print(f"\nInput rows    : {len(m1):,}")
print(f"Input columns : {len(m1.columns)}")

m1_required = [
    "location",
    "year",
    "soil_type",
    "annual_rainfall_mm",
    "temperature",
    "humidity",
    "soil_moisture",
    "crop"
]

check_required_columns(m1, m1_required, "M1")

# Keep ONLY model-required columns.
m1 = m1[m1_required].copy()

# Numeric conversion.
m1_numeric = [
    "year",
    "annual_rainfall_mm",
    "temperature",
    "humidity",
    "soil_moisture"
]

for col in m1_numeric:
    m1[col] = pd.to_numeric(m1[col], errors="coerce")

# Categorical cleaning.
m1_categorical = [
    "location",
    "soil_type",
    "crop"
]

for col in m1_categorical:
    m1[col] = (
        m1[col]
        .astype(str)
        .str.strip()
    )

check_no_missing(m1, "M1")

print("\nM1 target distribution:")
print(m1["crop"].value_counts())

# Stratified random split because this is a classification dataset.
m1_train, m1_test = train_test_split(
    m1,
    test_size=0.20,
    random_state=42,
    stratify=m1["crop"]
)

save_split(m1_train, m1_test, M1_OUT)

print("\nM1 completed.")
print(f"Total : {len(m1):,}")
print(f"Train : {len(m1_train):,}")
print(f"Test  : {len(m1_test):,}")


# ============================================================
# MODULE 2
# CONTEXT-AWARE IRRIGATION
# ============================================================

print("\n" + "=" * 70)
print("MODULE 2 - CONTEXT-AWARE IRRIGATION")
print("=" * 70)

check_file_exists(M2_FILE)

m2 = pd.read_csv(M2_FILE)
m2 = clean_column_names(m2)

print(f"\nInput rows    : {len(m2):,}")
print(f"Input columns : {len(m2.columns)}")

m2_required = [
    "timestamp",
    "air_temperature_c",
    "air_humidity_pct",
    "soil_temperature_c",
    "soil_moisture_pct",
    "water_pump_status"
]

check_required_columns(m2, m2_required, "M2")

# Keep timestamp temporarily for time-feature extraction.
m2 = m2[m2_required].copy()

m2["timestamp"] = normalize_datetime(
    m2["timestamp"],
    "timestamp",
    "M2"
)

# Extract useful time information.
m2["hour"] = m2["timestamp"].dt.hour
m2["day_of_week"] = m2["timestamp"].dt.dayofweek

# Raw timestamp is NOT a model feature.
m2 = m2.drop(columns=["timestamp"])

m2_numeric = [
    "air_temperature_c",
    "air_humidity_pct",
    "soil_temperature_c",
    "soil_moisture_pct",
    "hour",
    "day_of_week",
    "water_pump_status"
]

for col in m2_numeric:
    m2[col] = pd.to_numeric(m2[col], errors="coerce")

check_no_missing(m2, "M2")

print("\nM2 target distribution:")
print(m2["water_pump_status"].value_counts())

# Classification split.
m2_train, m2_test = train_test_split(
    m2,
    test_size=0.20,
    random_state=42,
    stratify=m2["water_pump_status"]
)

save_split(m2_train, m2_test, M2_OUT)

print("\nM2 completed.")
print(f"Total : {len(m2):,}")
print(f"Train : {len(m2_train):,}")
print(f"Test  : {len(m2_test):,}")


# ============================================================
# MODULE 3
# FUTURE SOIL MOISTURE PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("MODULE 3 - FUTURE SOIL MOISTURE PREDICTION")
print("=" * 70)

check_file_exists(M3_FILE)

m3 = pd.read_csv(M3_FILE)
m3 = clean_column_names(m3)

print(f"\nInput rows    : {len(m3):,}")
print(f"Input columns : {len(m3.columns)}")

m3_required = [
    "site",
    "timestamp",
    "air_temperature_c",
    "air_humidity_pct",
    "soil_temperature_c",
    "soil_moisture_pct",
    "moisture_lag_1h",
    "moisture_lag_3h",
    "rain_now_mm",
    "rain_sum_6h_mm",
    "rain_next_1h_mm",
    "hour_sin",
    "hour_cos",
    "target_moisture_1h"
]

check_required_columns(m3, m3_required, "M3")

m3 = m3[m3_required].copy()

m3["timestamp"] = normalize_datetime(
    m3["timestamp"],
    "timestamp",
    "M3"
)

# Sort chronologically within each site.
m3 = m3.sort_values(
    ["site", "timestamp"]
).reset_index(drop=True)

# IMPORTANT:
# rain_next_1h_mm is future information.
# It must NEVER be used as an input feature.
m3 = m3.drop(columns=["rain_next_1h_mm"])

m3_numeric = [
    "air_temperature_c",
    "air_humidity_pct",
    "soil_temperature_c",
    "soil_moisture_pct",
    "moisture_lag_1h",
    "moisture_lag_3h",
    "rain_now_mm",
    "rain_sum_6h_mm",
    "hour_sin",
    "hour_cos",
    "target_moisture_1h"
]

for col in m3_numeric:
    m3[col] = pd.to_numeric(m3[col], errors="coerce")

m3["site"] = (
    m3["site"]
    .astype(str)
    .str.strip()
)

check_no_missing(m3, "M3")

# Timestamp is retained only for chronological splitting.
# It is NOT included in the model features.
#
# The dataset already contains a future target, so we preserve
# the chronological structure instead of randomly shuffling.
unique_times = np.sort(m3["timestamp"].unique())

split_time_index = int(len(unique_times) * 0.80)
split_time = unique_times[split_time_index]

m3_train = m3[m3["timestamp"] < split_time].copy()
m3_test = m3[m3["timestamp"] >= split_time].copy()

# Timestamp is not needed by the model itself.
m3_train = m3_train.drop(columns=["timestamp"])
m3_test = m3_test.drop(columns=["timestamp"])

# Final M3 model columns.
m3_features = [
    "site",
    "air_temperature_c",
    "air_humidity_pct",
    "soil_temperature_c",
    "soil_moisture_pct",
    "moisture_lag_1h",
    "moisture_lag_3h",
    "rain_now_mm",
    "rain_sum_6h_mm",
    "hour_sin",
    "hour_cos"
]

m3_target = ["target_moisture_1h"]

m3_train = m3_train[m3_features + m3_target]
m3_test = m3_test[m3_features + m3_target]

save_split(m3_train, m3_test, M3_OUT)

print("\nM3 completed.")
print(f"Total : {len(m3):,}")
print(f"Train : {len(m3_train):,}")
print(f"Test  : {len(m3_test):,}")


# ============================================================
# MODULE 4
# RAINFALL FORECASTING
# ============================================================

print("\n" + "=" * 70)
print("MODULE 4 - HOURLY RAINFALL FORECASTING")
print("=" * 70)

check_file_exists(M4_FILE)

m4 = pd.read_csv(M4_FILE)
m4 = clean_column_names(m4)

print(f"\nInput rows    : {len(m4):,}")
print(f"Input columns : {len(m4.columns)}")

m4_required = [
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
    "wind_speed_kmh"
]

check_required_columns(m4, m4_required, "M4")

m4 = m4[m4_required].copy()

# ============================================================
# DATETIME
# ============================================================

m4["timestamp"] = normalize_datetime(
    m4["timestamp"],
    "timestamp",
    "M4"
)

# Sort by district and actual timestamp.
m4 = m4.sort_values(
    ["district", "timestamp"]
).reset_index(drop=True)

# ============================================================
# DUPLICATE OBSERVATIONS
# ============================================================

before_duplicates = len(m4)

m4 = m4.drop_duplicates(
    subset=["district", "timestamp"]
).reset_index(drop=True)

print(
    f"Duplicate district/timestamp rows removed: "
    f"{before_duplicates - len(m4):,}"
)

# ============================================================
# CREATE / VERIFY HOURLY TIME FEATURES
# ============================================================

m4["year"] = pd.to_numeric(m4["year"], errors="coerce")
m4["month"] = pd.to_numeric(m4["month"], errors="coerce")
m4["day_of_year"] = pd.to_numeric(m4["day_of_year"], errors="coerce")
m4["hour"] = pd.to_numeric(m4["hour"], errors="coerce")

# These features represent WHEN the hourly observation occurred.
# They are kept for model training.
#
# Raw date/time/timestamp remain only for processing.

# ============================================================
# NUMERIC WEATHER FEATURES
# ============================================================

m4_numeric = [
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "pressure_msl_hpa",
    "cloud_cover_percent",
    "wind_speed_kmh"
]

for col in m4_numeric:
    m4[col] = pd.to_numeric(m4[col], errors="coerce")

m4["district"] = (
    m4["district"]
    .astype(str)
    .str.strip()
)

check_no_missing(m4, "M4")

# ============================================================
# CREATE NEXT-HOUR RAINFALL TARGET
# ============================================================

m4["future_rainfall_mm"] = (
    m4.groupby("district")["rainfall_mm"]
    .shift(-1)
)

# Last observation of every district has no next-hour target.
m4 = m4.dropna(
    subset=["future_rainfall_mm"]
).reset_index(drop=True)

# ============================================================
# FINAL M4 MODEL FEATURES
# ============================================================

m4_features = [
    "district",
    "year",
    "month",
    "day_of_year",
    "hour",
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "pressure_msl_hpa",
    "cloud_cover_percent",
    "wind_speed_kmh"
]

m4_target = ["future_rainfall_mm"]

m4_model_data = m4[m4_features + m4_target].copy()

check_no_missing(m4_model_data, "M4 FINAL")

# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

# DO NOT randomly shuffle hourly forecasting data.
# Sort by actual time first.

m4_split_source = m4[
    ["district", "timestamp"]
].copy()

# Use a global chronological cutoff.
unique_times = np.sort(m4["timestamp"].unique())

split_time_index = int(len(unique_times) * 0.80)
split_time = unique_times[split_time_index]

train_mask = m4["timestamp"] < split_time
test_mask = m4["timestamp"] >= split_time

m4_train = m4.loc[
    train_mask,
    m4_features + m4_target
].copy()

m4_test = m4.loc[
    test_mask,
    m4_features + m4_target
].copy()

save_split(m4_train, m4_test, M4_OUT)

print("\nM4 completed.")
print(f"Total : {len(m4_model_data):,}")
print(f"Train : {len(m4_train):,}")
print(f"Test  : {len(m4_test):,}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nM1 - Crop Recommendation")
print(f"  Total : {len(m1):,}")
print(f"  Train : {len(m1_train):,}")
print(f"  Test  : {len(m1_test):,}")

print("\nM2 - Context-Aware Irrigation")
print(f"  Total : {len(m2):,}")
print(f"  Train : {len(m2_train):,}")
print(f"  Test  : {len(m2_test):,}")

print("\nM3 - Future Soil Moisture")
print(f"  Total : {len(m3):,}")
print(f"  Train : {len(m3_train):,}")
print(f"  Test  : {len(m3_test):,}")

print("\nM4 - Hourly Rainfall Forecasting")
print(f"  Total : {len(m4_model_data):,}")
print(f"  Train : {len(m4_train):,}")
print(f"  Test  : {len(m4_test):,}")

print("\nOutput directory:")
print(PREPROCESSED_DIR)

print("\nAll modules finished.")
