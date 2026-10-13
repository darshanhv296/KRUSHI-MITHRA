"""
FINAL TESTING SCRIPT - IOT FARMING
==================================
Sequential testing pipeline for models M1, M2, M3, and M4.
Models evaluated: Random Forest, XGBoost, LightGBM
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================================================
# PATHS
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # C:\...\IOT FARM\models
PROJECT_DIR = os.path.dirname(BASE_DIR)                # C:\...\IOT FARM

DATASET_DIR = os.path.join(PROJECT_DIR, "DATASET")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "PREPROCESSED")

# Paths aligned with train.py inside models/
MODEL_DIR = os.path.join(BASE_DIR, "TRAINED_MODELS")
RESULT_DIR = os.path.join(BASE_DIR, "TEST_RESULTS")

os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================================
# SETTINGS
# =====================================================================

BATCH_SIZE = 10000


# =====================================================================
# HELPERS
# =====================================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def model_path(module_name, model_name):
    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        + ".joblib"
    )
    return os.path.join(MODEL_DIR, module_name, filename)


def extract_pipeline(artifact):
    """Safely unpack artifact whether it is a raw pipeline or a dictionary."""
    if isinstance(artifact, dict):
        return artifact.get("pipeline", artifact.get("model", artifact))
    return artifact


def predict_with_progress(pipeline, X, model_name, task):
    """
    Run prediction in batches and display one progress bar.
    """
    total_rows = len(X)
    predictions = []
    start = time.time()

    progress = tqdm(
        total=total_rows,
        desc=f"    {model_name:<14}",
        unit="row",
        ncols=85,
        leave=True
    )

    try:
        for start_row in range(0, total_rows, BATCH_SIZE):
            end_row = min(start_row + BATCH_SIZE, total_rows)
            X_batch = X.iloc[start_row:end_row]
            batch_pred = pipeline.predict(X_batch)
            predictions.append(np.asarray(batch_pred))
            progress.update(end_row - start_row)
    finally:
        progress.close()

    elapsed = time.time() - start

    if predictions:
        predictions = np.concatenate(predictions)
    else:
        predictions = np.array([])

    return predictions, elapsed


# =====================================================================
# CLASSIFICATION TESTING
# =====================================================================

def test_classifier(module_name, test_df, features, target, model_names):
    X = test_df[features].copy().reset_index(drop=True)
    y = test_df[target].copy().astype(str).reset_index(drop=True)

    results = []

    print()
    print("=" * 75)
    print(f"{module_name} - CLASSIFICATION TEST")
    print("=" * 75)
    print(f"Rows  : {len(test_df):,}")
    print(f"Models: {len(model_names)}")

    for model_index, model_name in enumerate(model_names, start=1):
        path = model_path(module_name, model_name)
        print()
        print(f"[{model_index}/{len(model_names)}] Testing {model_name}")

        if not os.path.exists(path):
            print("    SKIPPED: model file not found:")
            print(f"    {path}")
            continue

        start_total = time.time()
        artifact = joblib.load(path)

        if isinstance(artifact, dict) and "pipeline" in artifact:
            pipeline = artifact["pipeline"]
            label_encoder = artifact.get("label_encoder")
        else:
            pipeline = extract_pipeline(artifact)
            label_encoder = None

        encoded_pred, prediction_time = predict_with_progress(
            pipeline, X, model_name, "classification"
        )

        if label_encoder is not None:
            pred = label_encoder.inverse_transform(
                np.round(encoded_pred).astype(int)
            ).astype(str)
        else:
            pred = encoded_pred.astype(str)

        accuracy = accuracy_score(y, pred)
        precision_weighted = precision_score(y, pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y, pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y, pred, average="weighted", zero_division=0)
        precision_macro = precision_score(y, pred, average="macro", zero_division=0)
        recall_macro = recall_score(y, pred, average="macro", zero_division=0)
        f1_macro = f1_score(y, pred, average="macro", zero_division=0)

        labels = sorted(set(y.unique()) | set(pred))
        cm = confusion_matrix(y, pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual_{x}" for x in labels],
            columns=[f"Predicted_{x}" for x in labels]
        )

        safe_name = model_name.lower().replace(" ", "_")
        cm_path = os.path.join(RESULT_DIR, f"{module_name}_{safe_name}_confusion_matrix.csv")
        cm_df.to_csv(cm_path)

        total_time = time.time() - start_total

        results.append({
            "module": module_name,
            "model": model_name,
            "task": "classification",
            "test_rows": len(test_df),
            "accuracy": accuracy,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "prediction_time_seconds": prediction_time,
            "total_test_time_seconds": total_time,
            "confusion_matrix_file": cm_path
        })

        print(
            f"    DONE | "
            f"Accuracy={accuracy:.6f} | "
            f"Weighted F1={f1_weighted:.6f} | "
            f"Macro F1={f1_macro:.6f}"
        )
        print(f"    Prediction time: {prediction_time:.2f}s")
        print(f"    Confusion matrix: {cm_path}")

    return results


# =====================================================================
# REGRESSION TESTING
# =====================================================================

def test_regressor(module_name, test_df, features, target, model_names):
    X = test_df[features].copy()
    y = pd.to_numeric(test_df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    results = []

    print()
    print("=" * 75)
    print(f"{module_name} - REGRESSION TEST")
    print("=" * 75)
    print(f"Rows  : {len(X):,}")
    print(f"Models: {len(model_names)}")

    for model_index, model_name in enumerate(model_names, start=1):
        path = model_path(module_name, model_name)
        print()
        print(f"[{model_index}/{len(model_names)}] Testing {model_name}")

        if not os.path.exists(path):
            print("    SKIPPED: model file not found:")
            print(f"    {path}")
            continue

        start_total = time.time()
        artifact = joblib.load(path)
        pipeline = extract_pipeline(artifact)

        pred, prediction_time = predict_with_progress(
            pipeline, X, model_name, "regression"
        )

        mae = mean_absolute_error(y, pred)
        model_rmse = rmse(y, pred)
        r2 = r2_score(y, pred)

        total_time = time.time() - start_total

        results.append({
            "module": module_name,
            "model": model_name,
            "task": "regression",
            "test_rows": len(X),
            "mae": mae,
            "rmse": model_rmse,
            "r2": r2,
            "prediction_time_seconds": prediction_time,
            "total_test_time_seconds": total_time
        })

        print(
            f"    DONE | "
            f"MAE={mae:.6f} | "
            f"RMSE={model_rmse:.6f} | "
            f"R2={r2:.6f}"
        )
        print(f"    Prediction time: {prediction_time:.2f}s")

    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    all_results = []
    total_start = time.time()

    print()
    print("=" * 75)
    print("IOT FARMING - MODEL TESTING")
    print("=" * 75)
    print(f"Prediction batch size: {BATCH_SIZE:,}")
    print("Testing mode: SEQUENTIAL")
    print("Progress: ONE BAR PER MODEL")

    active_models = ["Random Forest", "XGBoost", "LightGBM"]

    # =================================================================
    # M1 - CROP RECOMMENDATION
    # =================================================================
    m1_path = os.path.join(PREPROCESSED_DIR, "M1_Crop_Recommendation", "test.csv")
    if os.path.exists(m1_path):
        m1 = pd.read_csv(m1_path)
        m1_features = [
            "location", "soil_type", "year", "annual_rainfall_mm",
            "temperature", "humidity", "soil_moisture"
        ]
        all_results += test_classifier(
            "M1_Crop_Recommendation", m1, m1_features, "crop", active_models
        )
    else:
        print(f"\n[SKIP] Missing test set: {m1_path}")

    # =================================================================
    # M2 - CONTEXT-AWARE IRRIGATION
    # =================================================================
    m2_path = os.path.join(PREPROCESSED_DIR, "M2_Context_Aware_Irrigation", "test.csv")
    if os.path.exists(m2_path):
        m2 = pd.read_csv(m2_path)
        m2_features = [
            "air_temperature_c", "air_humidity_pct", "soil_temperature_c",
            "soil_moisture_pct", "hour", "day_of_week"
        ]
        all_results += test_classifier(
            "M2_Context_Aware_Irrigation", m2, m2_features, "water_pump_status", active_models
        )
    else:
        print(f"\n[SKIP] Missing test set: {m2_path}")

    # =================================================================
    # M3 - FUTURE SOIL MOISTURE
    # =================================================================
    m3_path = os.path.join(PREPROCESSED_DIR, "M3_Future_Soil_Moisture", "test.csv")
    if os.path.exists(m3_path):
        m3 = pd.read_csv(m3_path)
        m3_features = [
            "site", "air_temperature_c", "air_humidity_pct", "soil_temperature_c",
            "soil_moisture_pct", "moisture_lag_1h", "moisture_lag_3h",
            "rain_now_mm", "rain_sum_6h_mm", "hour_sin", "hour_cos"
        ]
        all_results += test_regressor(
            "M3_Future_Soil_Moisture", m3, m3_features, "target_moisture_1h", active_models
        )
    else:
        print(f"\n[SKIP] Missing test set: {m3_path}")

    # =================================================================
    # M4 - RAINFALL FORECASTING
    # =================================================================
    # Matches name in train.py ("M4_Rainfall_Forecasting")
    m4_path = os.path.join(PREPROCESSED_DIR, "M4_Rainfall_Forecasting", "test.csv")
    m4_folder = "M4_Rainfall_Forecasting"
    
    # Backward compatibility fallback if folder is still named M4_Hourly_Rainfall
    if not os.path.exists(m4_path):
        alt_path = os.path.join(PREPROCESSED_DIR, "M4_Hourly_Rainfall", "test.csv")
        if os.path.exists(alt_path):
            m4_path = alt_path
            m4_folder = "M4_Hourly_Rainfall"

    if os.path.exists(m4_path):
        m4 = pd.read_csv(m4_path)
        m4_features = [
            "district", "year", "month", "day_of_year", "hour",
            "rainfall_mm", "temperature_c", "humidity_percent",
            "pressure_msl_hpa", "cloud_cover_percent", "wind_speed_kmh"
        ]
        all_results += test_regressor(
            m4_folder, m4, m4_features, "rainfall_mm", active_models
        )
    else:
        print(f"\n[SKIP] Missing test set: {m4_path}")

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_path = os.path.join(RESULT_DIR, "final_test_results.csv")
        results_df.to_csv(summary_path, index=False)
        save_json(all_results, os.path.join(RESULT_DIR, "final_test_results.json"))

        classification_df = results_df[results_df["task"] == "classification"].copy()
        if not classification_df.empty:
            classification_df.to_csv(
                os.path.join(RESULT_DIR, "classification_test_results.csv"), index=False
            )

        regression_df = results_df[results_df["task"] == "regression"].copy()
        if not regression_df.empty:
            regression_df.to_csv(
                os.path.join(RESULT_DIR, "regression_test_results.csv"), index=False
            )
    else:
        summary_path = "(no tests run)"

    total_elapsed = time.time() - total_start

    print()
    print("=" * 75)
    print("ALL TESTING COMPLETED")
    print("=" * 75)
    print(f"Total testing time: {total_elapsed / 60:.2f} minutes")
    print(f"Final results: {summary_path}")

    if all_results:
        print()
        print("FINAL MODEL RESULTS")
        print("-" * 75)
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()