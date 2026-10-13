import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FOLDER = BASE_DIR


CLEANED_FOLDER = os.path.join(
    BASE_DIR,
    "CLEANED"
)

os.makedirs(
    CLEANED_FOLDER,
    exist_ok=True
)


# ============================================================
# FINAL DATASET FILES
# ============================================================

DATASET_FILES = [
    "Crop_Recommendation.csv",
    "Rain_Prediction.csv",
    "Smart_Irigation.csv",
    "Soil_Moisture_Prediction.csv"
]


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = {

    "Crop_Recommendation.csv": [
        "Location",
        "Year",
        "Soil_Type",
        "Annual_Rainfall_mm",
        "Temperature",
        "Humidity",
        "Soil_Moisture",
        "Crop"
    ],

    "Rain_Prediction.csv": [
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
    ],

    "Smart_Irigation.csv": [
        "timestamp",
        "air_temperature_c",
        "air_humidity_pct",
        "soil_temperature_c",
        "soil_moisture_pct",
        "water_pump_status"
    ],

    "Soil_Moisture_Prediction.csv": [
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
        "target_moisture_1h",
        "split"
    ]
}


# ============================================================
# HELPERS
# ============================================================

def validate_columns(df, filename):

    expected = EXPECTED_COLUMNS[filename]

    if list(df.columns) != expected:

        missing = [
            column
            for column in expected
            if column not in df.columns
        ]

        extra = [
            column
            for column in df.columns
            if column not in expected
        ]

        raise ValueError(
            f"\n{filename}: Dataset schema does not match "
            "the locked new dataset.\n"
            f"Missing columns: {missing}\n"
            f"Extra columns  : {extra}\n"
            f"Actual columns : {list(df.columns)}"
        )


def clean_column_names(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def clean_dataframe(df, filename):

    print("\n" + "-" * 70)
    print(f"Cleaning: {filename}")
    print("-" * 70)

    original_rows = len(df)
    original_columns = len(df.columns)

    # --------------------------------------------------------
    # Column names
    # --------------------------------------------------------

    df = clean_column_names(df)

    validate_columns(
        df,
        filename
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(f"Original rows    : {original_rows:,}")
    print(f"Original columns : {original_columns}")
    print(f"Duplicate rows   : {duplicate_count:,}")

    if duplicate_count > 0:

        df = (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

        print(
            f"Duplicates removed: "
            f"{duplicate_count:,}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isna().sum()
    total_missing = missing.sum()

    print("\nMissing values:")

    if total_missing == 0:

        print("  No missing values")

    else:

        print(
            missing[missing > 0]
        )

        # The locked new datasets are expected to contain
        # complete observations. Do not silently impute data.
        raise ValueError(
            f"\n{filename}: Missing values detected. "
            "Fix the source dataset before continuing."
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            f"\n{filename}: Dataset is empty after cleaning."
        )

    print("\nCleaning result:")
    print(f"Final rows : {len(df):,}")
    print(f"Final cols : {len(df.columns)}")
    print("STATUS     : CLEAN")

    return df


# ============================================================
# PROCESS ONE CSV
# ============================================================

def process_csv(filename):

    input_path = os.path.join(
        DATASET_FOLDER,
        filename
    )

    if not os.path.isfile(input_path):

        raise FileNotFoundError(
            f"\nDataset file not found:\n{input_path}"
        )

    print("\n" + "=" * 70)
    print(f"PROCESSING: {filename}")
    print("=" * 70)

    df = pd.read_csv(
        input_path
    )

    print(f"\nInput:")
    print(input_path)

    cleaned_df = clean_dataframe(
        df,
        filename
    )

    output_name = (
        os.path.splitext(filename)[0]
        + "_cleaned.csv"
    )

    output_path = os.path.join(
        CLEANED_FOLDER,
        output_name
    )

    cleaned_df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)


# ============================================================
# MAIN CLEANER
# ============================================================

def clean_all_datasets():

    print("\n" + "=" * 70)
    print("IOT FARMING - FINAL DATASET CLEANER")
    print("=" * 70)

    print("\nInput folder:")
    print(DATASET_FOLDER)

    print("\nOutput folder:")
    print(CLEANED_FOLDER)

    print("\nFinal datasets:")

    for number, filename in enumerate(
        DATASET_FILES,
        start=1
    ):

        print(
            f"{number}. {filename}"
        )

    for filename in DATASET_FILES:

        try:

            process_csv(
                filename
            )

        except Exception as error:

            print("\n" + "!" * 70)
            print("ERROR")
            print("!" * 70)
            print(f"File: {filename}")
            print(repr(error))

            raise

    print("\n" + "=" * 70)
    print("ALL FINAL DATASETS CLEANED SUCCESSFULLY")
    print("=" * 70)

    print("\nCleaned datasets:")
    print(CLEANED_FOLDER)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    clean_all_datasets()
