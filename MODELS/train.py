"""
FINAL TRAINING SCRIPT - IOT FARMING
====================================
Sequential training pipeline for modules M1, M2, M3 and M4.

Models per module: Random Forest, XGBoost, LightGBM (SVM / SVR removed)

Progress bars (real, per-iteration):
  - Random Forest : advances per batch of trees (warm_start)
  - XGBoost       : advances per boosting round (TrainingCallback)
  - LightGBM      : advances per boosting round (callback)

Evaluation:
  - If DATASET/PREPROCESSED/<module>/test.csv exists, it is used as the holdout set.
  - Otherwise an 80/20 holdout split is made from train.csv.
  Both train and holdout metrics are reported, so overfitting is visible.

Saved artifact format is unchanged:
  {"pipeline": Pipeline, "label_encoder": LabelEncoder (classifiers only), "features": [...]}
"""

import os
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback

warnings.filterwarnings("ignore")

# =====================================================================
# PATHS
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # C:\...\IOT FARM\models
PROJECT_DIR = os.path.dirname(BASE_DIR)                # C:\...\IOT FARM
DATASET_DIR = os.path.join(PROJECT_DIR, "DATASET")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "PREPROCESSED")

# Stored directly in models/ folder
MODEL_DIR = os.path.join(BASE_DIR, "TRAINED_MODELS")
RESULT_DIR = os.path.join(BASE_DIR, "TRAINING_RESULTS")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================================
# GLOBAL SETTINGS
# =====================================================================

RANDOM_STATE = 42
N_JOBS = -1
HOLDOUT_FRACTION = 0.20
N_ESTIMATORS = 200
RF_PROGRESS_STEPS = 20  # how many times the Random Forest bar updates

# =====================================================================
# LIGHTGBM CHECK
# =====================================================================

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("\nWARNING: LightGBM is not installed (pip install lightgbm). "
          "Training will continue without it.\n")

# =====================================================================
# HELPERS
# =====================================================================


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(categorical_features, numeric_features):
    transformers = []

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if numeric_features:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("num", numeric_pipeline, numeric_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_model_filename(model_name):
    return model_name.lower().replace(" ", "_").replace("/", "_")


def make_bar(model_name, total, unit):
    return tqdm(total=total, desc=f"    {model_name:<14}", unit=unit, ncols=90, leave=True)


# =====================================================================
# PROGRESS CALLBACKS
# =====================================================================


class XGBoostProgressBar(TrainingCallback):
    """Advances the bar once per boosting round."""

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def after_iteration(self, model, epoch, evals_log):
        self._pbar.update(epoch + 1 - self._pbar.n)
        return False


class LightGBMProgressBar:
    """Advances the bar once per boosting round."""

    order = 10
    before_iteration = False

    def __init__(self, pbar):
        self._pbar = pbar

    def __call__(self, env):
        self._pbar.update(env.iteration + 1 - self._pbar.n)


# =====================================================================
# FITTING
# =====================================================================


def fit_model(pipeline, X, y, model_name):
    """
    Fits the preprocessor once, then fits the model with a real progress bar.
    Returns (fitted pipeline, elapsed seconds).
    """
    start = time.time()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_trans = preprocessor.fit_transform(X)

    if model_name == "Random Forest":
        total = model.n_estimators
        step = max(1, total // RF_PROGRESS_STEPS)
        model.set_params(warm_start=True)
        try:
            with make_bar(model_name, total, "tree") as pbar:
                current = 0
                while current < total:
                    current = min(current + step, total)
                    model.set_params(n_estimators=current)
                    model.fit(X_trans, y)
                    pbar.update(current - pbar.n)
        finally:
            model.set_params(warm_start=False)

    elif model_name == "XGBoost":
        with make_bar(model_name, model.n_estimators, "round") as pbar:
            try:
                model.set_params(callbacks=[XGBoostProgressBar(pbar)])
                model.fit(X_trans, y)
                pbar.update(model.n_estimators - pbar.n)
            finally:
                # The callback holds the tqdm bar (not picklable); remove before saving.
                model.set_params(callbacks=None)

    elif model_name == "LightGBM":
        with make_bar(model_name, model.n_estimators, "round") as pbar:
            model.fit(X_trans, y, callbacks=[LightGBMProgressBar(pbar)])
            pbar.update(model.n_estimators - pbar.n)

            # Strip callbacks to avoid pickling issues
            if hasattr(model, "callbacks_"):
                model.callbacks_ = []
            if hasattr(model, "callbacks"):
                model.callbacks = None

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return pipeline, time.time() - start


# =====================================================================
# DATA SPLITTING
# =====================================================================


def load_train_and_holdout(module_name, df, target, task):
    """
    Returns (train_df, holdout_df, holdout_source).
    Uses test.csv next to train.csv when present, else a random holdout split.
    """
    df = df.dropna(subset=[target]).reset_index(drop=True)

    test_path = os.path.join(PREPROCESSED_DIR, module_name, "test.csv")
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path).dropna(subset=[target]).reset_index(drop=True)
        return df, test_df, "test.csv"

    stratify = None
    if task == "classifier":
        counts = df[target].astype(str).value_counts()
        if counts.min() >= 2:
            stratify = df[target].astype(str)

    train_df, hold_df = train_test_split(
        df, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE, stratify=stratify
    )
    return train_df.reset_index(drop=True), hold_df.reset_index(drop=True), f"{int(HOLDOUT_FRACTION * 100)}% split"


# =====================================================================
# CLASSIFICATION ENGINE
# =====================================================================


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_classifier(module_name, df, features, target, cat_features, num_features, model_specs):
    module_model_dir = os.path.join(MODEL_DIR, module_name)
    os.makedirs(module_model_dir, exist_ok=True)

    train_df, hold_df, hold_src = load_train_and_holdout(module_name, df, target, "classifier")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[target].astype(str))

    if len(label_encoder.classes_) < 2:
        print(f"[SKIP] {module_name}: target '{target}' has fewer than 2 classes.")
        return []

    known = hold_df[target].astype(str).isin(label_encoder.classes_)
    if (~known).any():
        print(f"Note: dropped {int((~known).sum())} holdout rows with unseen classes.")
    hold_df = hold_df.loc[known].reset_index(drop=True)
    y_hold = label_encoder.transform(hold_df[target].astype(str))

    X_train = train_df[features].reset_index(drop=True)
    X_hold = hold_df[features].reset_index(drop=True)

    print("\n" + "=" * 75)
    print(f"{module_name} - CLASSIFICATION")
    print("=" * 75)
    print(f"Train rows: {len(X_train):,} | Holdout rows: {len(X_hold):,} ({hold_src}) | "
          f"Target: {target} | Classes: {len(label_encoder.classes_)}")

    results = []
    for idx, (model_name, model_inst) in enumerate(model_specs, start=1):
        print(f"\n[{idx}/{len(model_specs)}] Training: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(cat_features, num_features)),
            ("model", model_inst),
        ])

        pipeline, elapsed = fit_model(pipeline, X_train, y_train, model_name)

        train_m = classification_metrics(y_train, pipeline.predict(X_train))
        hold_m = classification_metrics(y_hold, pipeline.predict(X_hold)) if len(X_hold) else {}

        model_path = os.path.join(module_model_dir, f"{safe_model_filename(model_name)}.joblib")
        joblib.dump({
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "features": features,
        }, model_path)

        row = {
            "module": module_name,
            "model": model_name,
            "task": "classification",
            "train_rows": len(X_train),
            "holdout_rows": len(X_hold),
            "holdout_source": hold_src,
            "training_time_seconds": elapsed,
            "model_file": model_path,
        }
        row.update({f"{k}_train": v for k, v in train_m.items()})
        row.update({f"{k}_holdout": v for k, v in hold_m.items()})
        results.append(row)

        print(f"    Train   -> Acc: {train_m['accuracy']:.4f} | F1: {train_m['f1_weighted']:.4f}")
        if hold_m:
            print(f"    Holdout -> Acc: {hold_m['accuracy']:.4f} | F1: {hold_m['f1_weighted']:.4f}")
        print(f"    Time: {elapsed:.2f}s | Saved -> {model_path}")

    return results


# =====================================================================
# REGRESSION ENGINE
# =====================================================================


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_regressor(module_name, df, features, target, cat_features, num_features, model_specs):
    module_model_dir = os.path.join(MODEL_DIR, module_name)
    os.makedirs(module_model_dir, exist_ok=True)

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)

    train_df, hold_df, hold_src = load_train_and_holdout(module_name, df, target, "regressor")
    hold_df[target] = pd.to_numeric(hold_df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    hold_df = hold_df.dropna(subset=[target]).reset_index(drop=True)

    X_train = train_df[features].reset_index(drop=True)
    y_train = train_df[target].values
    X_hold = hold_df[features].reset_index(drop=True)
    y_hold = hold_df[target].values

    print("\n" + "=" * 75)
    print(f"{module_name} - REGRESSION")
    print("=" * 75)
    print(f"Train rows: {len(X_train):,} | Holdout rows: {len(X_hold):,} ({hold_src}) | Target: {target}")

    results = []
    for idx, (model_name, model_inst) in enumerate(model_specs, start=1):
        print(f"\n[{idx}/{len(model_specs)}] Training: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(cat_features, num_features)),
            ("model", model_inst),
        ])

        pipeline, elapsed = fit_model(pipeline, X_train, y_train, model_name)

        train_m = regression_metrics(y_train, pipeline.predict(X_train))
        hold_m = regression_metrics(y_hold, pipeline.predict(X_hold)) if len(X_hold) else {}

        model_path = os.path.join(module_model_dir, f"{safe_model_filename(model_name)}.joblib")
        joblib.dump({"pipeline": pipeline, "features": features}, model_path)

        row = {
            "module": module_name,
            "model": model_name,
            "task": "regression",
            "train_rows": len(X_train),
            "holdout_rows": len(X_hold),
            "holdout_source": hold_src,
            "training_time_seconds": elapsed,
            "model_file": model_path,
        }
        row.update({f"{k}_train": v for k, v in train_m.items()})
        row.update({f"{k}_holdout": v for k, v in hold_m.items()})
        results.append(row)

        print(f"    Train   -> MAE: {train_m['mae']:.4f} | RMSE: {train_m['rmse']:.4f} | R2: {train_m['r2']:.4f}")
        if hold_m:
            print(f"    Holdout -> MAE: {hold_m['mae']:.4f} | RMSE: {hold_m['rmse']:.4f} | R2: {hold_m['r2']:.4f}")
        print(f"    Time: {elapsed:.2f}s | Saved -> {model_path}")

    return results


# =====================================================================
# MODEL SPECIFICATIONS (Random Forest, XGBoost, LightGBM)
# =====================================================================


def get_classifier_models():
    models = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ("XGBoost", XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=6, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", random_state=RANDOM_STATE, n_jobs=N_JOBS)),
    ]
    if LIGHTGBM_AVAILABLE:
        models.append(("LightGBM", LGBMClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.08, num_leaves=31,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)))
    return models


def get_regressor_models():
    models = [
        ("Random Forest", RandomForestRegressor(
            n_estimators=N_ESTIMATORS, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ("XGBoost", XGBRegressor(
            n_estimators=N_ESTIMATORS, max_depth=6, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", random_state=RANDOM_STATE, n_jobs=N_JOBS)),
    ]
    if LIGHTGBM_AVAILABLE:
        models.append(("LightGBM", LGBMRegressor(
            n_estimators=N_ESTIMATORS, learning_rate=0.06, num_leaves=31,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)))
    return models


# =====================================================================
# MAIN PIPELINE
# =====================================================================


def main():
    all_results = []
    total_start = time.time()

    print("=" * 75)
    print("IOT FARMING - MODEL TRAINING PIPELINE")
    print("=" * 75)

    modules_config = [
        {
            "name": "M1_Crop_Recommendation",
            "type": "classifier",
            "target": "crop",
            "cat_features": ["location", "soil_type"],
            "num_features": ["year", "annual_rainfall_mm", "temperature", "humidity", "soil_moisture"],
            "models": get_classifier_models,
        },
        {
            "name": "M2_Context_Aware_Irrigation",
            "type": "classifier",
            "target": "water_pump_status",
            "cat_features": [],
            "num_features": ["air_temperature_c", "air_humidity_pct", "soil_temperature_c",
                             "soil_moisture_pct", "hour", "day_of_week"],
            "models": get_classifier_models,
        },
        {
            "name": "M3_Future_Soil_Moisture",
            "type": "regressor",
            "target": "target_moisture_1h",
            "cat_features": ["site"],
            "num_features": ["air_temperature_c", "air_humidity_pct", "soil_temperature_c", "soil_moisture_pct",
                             "moisture_lag_1h", "moisture_lag_3h", "rain_now_mm", "rain_sum_6h_mm",
                             "hour_sin", "hour_cos"],
            "models": get_regressor_models,
        },
        {
            "name": "M4_Rainfall_Forecasting",
            "type": "regressor",
            "target": "future_rainfall_mm",
            "cat_features": ["district"],
            "num_features": ["year", "month", "day_of_year", "hour", "rainfall_mm", "temperature_c",
                             "humidity_percent", "pressure_msl_hpa", "cloud_cover_percent", "wind_speed_kmh"],
            "models": get_regressor_models,
        },
    ]

    for cfg in modules_config:
        csv_path = os.path.join(PREPROCESSED_DIR, cfg["name"], "train.csv")
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] Missing dataset file: {csv_path}")
            continue

        print(f"\nProcessing module: {cfg['name']}")
        df = pd.read_csv(csv_path)
        features = cfg["cat_features"] + cfg["num_features"]

        missing = [c for c in features + [cfg["target"]] if c not in df.columns]
        if missing:
            print(f"[SKIP] {cfg['name']}: columns missing in train.csv: {missing}")
            continue

        trainer = train_classifier if cfg["type"] == "classifier" else train_regressor
        all_results.extend(trainer(
            cfg["name"], df, features, cfg["target"],
            cfg["cat_features"], cfg["num_features"], cfg["models"](),
        ))

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(RESULT_DIR, "training_summary.csv")
        results_df.to_csv(results_path, index=False)
        with open(os.path.join(RESULT_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, default=str)
    else:
        results_path = "(nothing trained)"

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print("ALL TRAINING COMPLETE")
    print(f"Total time elapsed: {total_elapsed / 60:.2f} minutes")
    print(f"Results saved to  : {results_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()