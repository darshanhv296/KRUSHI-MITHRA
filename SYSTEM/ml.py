"""
Krushi Mithra
Smart Farming & Intelligent Irrigation System

ml.py
=====
Central ML inference layer for the trained model artifacts.
"""

from __future__ import annotations

import logging
import json
import math
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from config import DATA_DIR, MODEL_DIR
from weather import (
    M1WeatherHistory,
    M4WeatherHistory,
    PRIMARY_REGION,
    get_current_weather_for_location,
    get_current_weather_for_region,
    get_m1_annual_rainfall_reference,
    get_m3_rain_sum_6h,
)

logger = logging.getLogger("KrushiMithra.ML")

M1_EXPECTED_FEATURES = [
    "location",
    "soil_type",
    "year",
    "annual_rainfall_mm",
    "temperature",
    "humidity",
    "soil_moisture",
]

M2_EXPECTED_FEATURES = [
    "air_temperature_c",
    "air_humidity_pct",
    "soil_temperature_c",
    "soil_moisture_pct",
    "hour",
    "day_of_week",
]

M3_EXPECTED_FEATURES = [
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
    "hour_cos",
]

M4_EXPECTED_FEATURES = [
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
    "wind_speed_kmh",
]


class MLError(Exception):
    """Base ML exception."""


class ModelLoadError(MLError):
    """Raised when a trained model cannot be loaded."""


class PredictionError(MLError):
    """Raised when prediction cannot be generated."""


class FeatureError(MLError):
    """Raised when required model features are missing or invalid."""


class HistoryUnavailableError(FeatureError):
    """Raised when M3 lacks real historical moisture readings."""


def _resolve_model_file(directory: Path, filename: str) -> Path:
    if not directory.exists():
        raise ModelLoadError(f"Model directory not found: {directory}")

    target = filename.lower()
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.name.lower() == target:
            return candidate

    raise ModelLoadError(f"Model file not found: {directory / filename}")


class MLEngine:
    """Central inference engine for the trained Krushi Mithra models."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.m1_model = None
        self.m2_model = None
        self.m3_model = None
        self.m4_model = None
        self.m1_weather = M1WeatherHistory()
        self.m4_weather = M4WeatherHistory()
        self._m3_history: deque[Tuple[datetime, float]] = deque(maxlen=600)
        self._m3_history_lock = threading.RLock()
        self._m3_history_path = DATA_DIR / "m3_sensor_history.jsonl"
        self._load_m3_history()
        self.load_models()

    def _rewrite_m3_history(self) -> None:
        temporary_path = self._m3_history_path.with_suffix(".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as history_file:
                for timestamp, moisture in self._m3_history:
                    history_file.write(json.dumps({
                        "timestamp": timestamp.isoformat(),
                        "soil_moisture_percent": moisture,
                    }) + "\n")
            temporary_path.replace(self._m3_history_path)
        except OSError as exc:
            logger.warning("Could not persist M3 sensor history: %s", exc)

    def _load_m3_history(self) -> None:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        cutoff = now - timedelta(hours=5)
        readings = []
        try:
            with self._m3_history_path.open("r", encoding="utf-8") as history_file:
                for line in history_file:
                    try:
                        record = json.loads(line)
                        timestamp = datetime.fromisoformat(record["timestamp"])
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
                        timestamp = timestamp.astimezone(ZoneInfo("Asia/Kolkata"))
                        moisture = float(record["soil_moisture_percent"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if cutoff <= timestamp <= now and math.isfinite(moisture) and 0.0 <= moisture <= 100.0:
                        readings.append((timestamp, moisture))
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Could not load M3 sensor history: %s", exc)
            return

        readings.sort(key=lambda reading: reading[0])
        with self._m3_history_lock:
            self._m3_history.extend(readings[-self._m3_history.maxlen:])
        self._rewrite_m3_history()
        if readings:
            logger.info("Loaded %d recent M3 sensor readings from disk.", len(self._m3_history))

    @staticmethod
    def _validate_required(data: Dict[str, Any], required: Sequence[str], model_name: str) -> None:
        missing = [name for name in required if name not in data or data[name] is None]
        if missing:
            raise FeatureError(f"{model_name}: missing features: {missing}")

    @staticmethod
    def _require_float(value: Any, name: str) -> float:
        if value is None:
            raise FeatureError(f"{name} is required.")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise FeatureError(f"{name} is invalid: {value!r}") from exc

    @staticmethod
    def _require_finite_float(value: Any, name: str) -> float:
        result = MLEngine._require_float(value, name)
        if not math.isfinite(result):
            raise FeatureError(f"{name} must be finite.")
        return result

    @staticmethod
    def soil_raw_to_percent(raw: float) -> float:
        raw = float(raw)
        dry = 4095.0
        wet = 1500.0
        if raw >= dry:
            return 0.0
        if raw <= wet:
            return 100.0
        percentage = ((dry - raw) / (dry - wet)) * 100.0
        return max(0.0, min(100.0, percentage))

    def _load_artifact(self, module_name: str, preferred_name: str, expected_features: Sequence[str], require_label_encoder: bool = False):
        module_dir = self.model_dir / module_name
        path = _resolve_model_file(module_dir, preferred_name)
        logger.info("[ML] Loading %s from %s", module_name, path)

        try:
            artifact = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model artifact: {path} | Reason: {exc}") from exc

        if not isinstance(artifact, dict):
            raise ModelLoadError(f"Model artifact is not a dict: {path}")
        if "pipeline" not in artifact:
            raise ModelLoadError(f"Missing 'pipeline' in artifact: {path}")

        feature_list = list(artifact.get("features", []))
        if feature_list != list(expected_features):
            raise FeatureError(
                f"{module_name} feature mismatch: expected {list(expected_features)} but found {feature_list}."
            )

        if require_label_encoder and "label_encoder" not in artifact:
            raise ModelLoadError(f"Missing 'label_encoder' in classifier artifact: {path}")

        return artifact

    def load_models(self) -> None:
        logger.info("[ML] Loading M1 Random Forest...")
        self.m1_model = self._load_artifact(
            "M1_Crop_Recommendation",
            "random_forest.joblib",
            M1_EXPECTED_FEATURES,
            require_label_encoder=True,
        )
        logger.info("[ML] M1 loaded successfully")

        logger.info("[ML] Loading M2 LightGBM...")
        self.m2_model = self._load_artifact(
            "M2_Context_Aware_Irrigation",
            "lightgbm.joblib",
            M2_EXPECTED_FEATURES,
            require_label_encoder=True,
        )
        logger.info("[ML] M2 loaded successfully")

        logger.info("[ML] Loading M3 LightGBM...")
        self.m3_model = self._load_artifact(
            "M3_Future_Soil_Moisture",
            "lightgbm.joblib",
            M3_EXPECTED_FEATURES,
        )
        logger.info("[ML] M3 loaded successfully")

        logger.info("[ML] Loading M4 XGBoost...")
        m4_dir = self.model_dir / "M4_Rainfall_Forecasting"
        if not m4_dir.exists():
            m4_dir = self.model_dir / "M4_Hourly_Rainfall"
        self.m4_model = self._load_artifact(
            m4_dir.name,
            "xgboost.joblib",
            M4_EXPECTED_FEATURES,
        )
        logger.info("[ML] M4 loaded successfully")

    def _require_loaded_model(self, artifact: Optional[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        if artifact is None:
            raise ModelLoadError(f"{model_name} model is not loaded.")
        return artifact

    def _get_pipeline(self, artifact: Dict[str, Any]):
        pipeline = artifact.get("pipeline")
        if pipeline is None:
            raise ModelLoadError("Artifact is missing the trained pipeline.")
        return pipeline

    @staticmethod
    def _get_categorical_values(artifact: Dict[str, Any], feature: str) -> list[str]:
        pipeline = artifact.get("pipeline")
        if pipeline is None:
            raise ModelLoadError("Artifact is missing the trained pipeline.")

        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            raise ModelLoadError("Trained pipeline is missing its preprocessor.")

        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name != "cat" or feature not in columns:
                continue

            encoder = transformer.named_steps.get("onehot")
            if encoder is None:
                raise ModelLoadError("Categorical preprocessor is missing its encoder.")

            index = columns.index(feature)
            return sorted(str(value) for value in encoder.categories_[index])

        raise FeatureError(
            f"Categorical feature is not present in the trained model: {feature}."
        )

    def prediction_options(self) -> Dict[str, list[str]]:
        m1 = self._require_loaded_model(self.m1_model, "M1")
        m4 = self._require_loaded_model(self.m4_model, "M4")
        return {
            "crop_locations": self._get_categorical_values(m1, "location"),
            "soil_types": self._get_categorical_values(m1, "soil_type"),
            "rainfall_regions": self._get_categorical_values(m4, "district"),
        }

    def _predict_with_pipeline(self, artifact: Dict[str, Any], frame: pd.DataFrame):
        return self._get_pipeline(artifact).predict(frame)

    def _history_annual_rainfall_mm(self, region: str, reference_date: datetime) -> float:
        return float(self.m1_weather.rolling_365d_rainfall(region=region, reference_time=reference_date))

    def prepare_m1_input(
        self,
        location: str,
        soil_type: str,
        year: Optional[int] = None,
        annual_rainfall_mm: Optional[float] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        soil_moisture: Optional[float] = None,
        reference_date: Optional[datetime] = None,
        region: str = PRIMARY_REGION,
        irrigation: Optional[str] = None,
    ) -> pd.DataFrame:
        if reference_date is None:
            reference_date = datetime.now()
        if year is None:
            year = int(reference_date.year)
        if annual_rainfall_mm is None:
            annual_rainfall_mm = self._history_annual_rainfall_mm(region, reference_date)
        if temperature is None or humidity is None or soil_moisture is None:
            raise FeatureError("M1 crop prediction requires temperature, humidity, and soil_moisture.")

        data = {
            "location": str(location),
            "soil_type": str(soil_type),
            "year": int(year),
            "annual_rainfall_mm": float(annual_rainfall_mm),
            "temperature": float(temperature),
            "humidity": float(humidity),
            "soil_moisture": float(soil_moisture),
        }
        self._validate_required(data, M1_EXPECTED_FEATURES, "M1")
        return pd.DataFrame([data], columns=M1_EXPECTED_FEATURES)

    def predict_m1(
        self,
        location: str,
        soil_type: str,
        irrigation: Optional[str] = None,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        reference_date: Optional[datetime] = None,
        region: str = PRIMARY_REGION,
        annual_rainfall_mm: Optional[float] = None,
        soil_moisture: Optional[float] = None,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            frame = self.prepare_m1_input(
                location=location,
                soil_type=soil_type,
                year=year,
                annual_rainfall_mm=annual_rainfall_mm,
                temperature=temperature,
                humidity=humidity,
                soil_moisture=soil_moisture,
                reference_date=reference_date,
                region=region,
                irrigation=irrigation,
            )
            model = self._require_loaded_model(self.m1_model, "M1")
            prediction = self._predict_with_pipeline(model, frame)
            label_encoder = model.get("label_encoder")
            if label_encoder is not None:
                prediction_label = str(
                    label_encoder.inverse_transform([int(round(float(prediction[0])))])[0]
                )
            else:
                prediction_label = str(prediction[0])
            result = {
                "success": True,
                "module": "M1_Crop_Recommendation",
                "model": "Random Forest",
                "crop": prediction_label,
                "features": frame.iloc[0].to_dict(),
            }

            pipeline = self._get_pipeline(model)
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(frame)[0]
                classes = label_encoder.inverse_transform(list(range(len(probabilities)))) if label_encoder is not None else list(range(len(probabilities)))
                confidence = float(max(probabilities))
                result["confidence"] = confidence
                result["class_probabilities"] = {str(c): float(p) for c, p in zip(classes, probabilities)}

            logger.info("[M1] Crop: %s | Confidence: %.2f%%", result.get("crop"), (result.get("confidence", 0.0) * 100.0))
            return result
        except Exception as exc:
            logger.exception("[M1] prediction failed: %s", exc)
            return {"success": False, "module": "M1_Crop_Recommendation", "error": str(exc)}

    def prepare_m2_input(
        self,
        esp32_data: Dict[str, Any],
        current_datetime: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if current_datetime is None:
            current_datetime = datetime.now()

        air_temp = esp32_data.get("air_temperature_c", esp32_data.get("temperature"))
        air_humidity = esp32_data.get("air_humidity_pct", esp32_data.get("humidity"))
        soil_temp = esp32_data.get("soil_temperature_c", esp32_data.get("soil_temperature"))
        soil_moisture = esp32_data.get("soil_moisture_pct", esp32_data.get("soil_moisture_percent"))
        if soil_moisture is None and esp32_data.get("soil_moisture_raw") is not None:
            soil_moisture = self.soil_raw_to_percent(float(esp32_data["soil_moisture_raw"]))

        if air_temp is None or air_humidity is None or soil_temp is None or soil_moisture is None:
            raise FeatureError("M2 requires air_temperature_c, air_humidity_pct, soil_temperature_c, and soil_moisture_pct.")

        air_temp_val = self._require_float(air_temp, "M2 air_temperature_c")
        air_humidity_val = self._require_float(air_humidity, "M2 air_humidity_pct")
        soil_temp_val = self._require_float(soil_temp, "M2 soil_temperature_c")
        soil_moisture_val = self._require_float(soil_moisture, "M2 soil_moisture_pct")

        self._validate_required(
            {
                "air_temperature_c": air_temp_val,
                "air_humidity_pct": air_humidity_val,
                "soil_temperature_c": soil_temp_val,
                "soil_moisture_pct": soil_moisture_val,
                "hour": current_datetime.hour,
                "day_of_week": current_datetime.weekday(),
            },
            M2_EXPECTED_FEATURES,
            "M2",
        )

        data = {
            "air_temperature_c": air_temp_val,
            "air_humidity_pct": air_humidity_val,
            "soil_temperature_c": soil_temp_val,
            "soil_moisture_pct": soil_moisture_val,
            "hour": int(current_datetime.hour),
            "day_of_week": int(current_datetime.weekday()),
        }
        return pd.DataFrame([data], columns=M2_EXPECTED_FEATURES)

    def predict_m2(
        self,
        esp32_data: Dict[str, Any],
        crop_type: Optional[str] = None,
        best_time: str = "Any",
        current_datetime: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        try:
            frame = self.prepare_m2_input(esp32_data, current_datetime=current_datetime)
            model = self._require_loaded_model(self.m2_model, "M2")
            prediction = self._predict_with_pipeline(model, frame)
            value = int(prediction[0])
            result = {
                "success": True,
                "module": "M2_Context_Aware_Irrigation",
                "model": "LightGBM",
                "water_pump_status": value,
                "pump_status": value,
                "irrigation": "ON" if value else "OFF",
                "features": frame.iloc[0].to_dict(),
            }
            pipeline = self._get_pipeline(model)
            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(frame)[0]
                label_encoder = model.get("label_encoder")
                classes = label_encoder.classes_ if label_encoder is not None else [0, 1]
                confidence = float(max(probabilities))
                result["confidence"] = confidence
                result["class_probabilities"] = {str(c): float(p) for c, p in zip(classes, probabilities)}
            logger.info("[M2] Pump prediction: %s | Confidence: %.2f%%", "ON" if value else "OFF", (result.get("confidence", 0.0) * 100.0))
            return result
        except Exception as exc:
            logger.exception("[M2] prediction failed: %s", exc)
            return {"success": False, "module": "M2_Context_Aware_Irrigation", "error": str(exc)}

    def _update_m3_history(self, soil_moisture: float, timestamp: datetime) -> bool:
        soil_moisture = float(soil_moisture)
        if not math.isfinite(soil_moisture) or not 0.0 <= soil_moisture <= 100.0:
            return False

        normalized = pd.Timestamp(timestamp)
        if normalized.tzinfo is None:
            normalized = normalized.tz_localize("Asia/Kolkata")
        else:
            normalized = normalized.tz_convert("Asia/Kolkata")
        normalized_datetime = normalized.to_pydatetime()

        with self._m3_history_lock:
            if self._m3_history:
                previous_timestamp, _ = self._m3_history[-1]
                elapsed = (normalized_datetime - previous_timestamp).total_seconds()
                if elapsed <= 0:
                    return False
                if elapsed < 30:
                    return False

            self._m3_history.append((normalized_datetime, soil_moisture))
            cutoff = normalized_datetime - timedelta(hours=5)
            while self._m3_history and self._m3_history[0][0] < cutoff:
                self._m3_history.popleft()
            try:
                with self._m3_history_path.open("a", encoding="utf-8") as history_file:
                    history_file.write(json.dumps({
                        "timestamp": normalized_datetime.isoformat(),
                        "soil_moisture_percent": soil_moisture,
                    }) + "\n")
                if self._m3_history_path.stat().st_size > 262144:
                    self._rewrite_m3_history()
            except OSError as exc:
                logger.warning("Could not persist M3 sensor reading: %s", exc)
            return True

    def record_m3_reading(
        self,
        esp32_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Record a valid live sensor sample for later M3 lag features."""
        moisture = esp32_data.get("soil_moisture_pct", esp32_data.get("soil_moisture_percent"))
        if moisture is None and esp32_data.get("soil_moisture_raw") is not None:
            try:
                moisture = self.soil_raw_to_percent(float(esp32_data["soil_moisture_raw"]))
            except (TypeError, ValueError):
                return False

        try:
            moisture_value = self._require_finite_float(moisture, "M3 soil_moisture_pct")
        except FeatureError:
            return False
        if not 0.0 <= moisture_value <= 100.0:
            return False

        return self._update_m3_history(
            moisture_value,
            timestamp or datetime.now(ZoneInfo("Asia/Kolkata")),
        )

    def _nearest_past_value(self, current_datetime: datetime, target_hours: float) -> Optional[float]:
        current_ts = pd.Timestamp(current_datetime)
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("Asia/Kolkata")
        else:
            current_ts = current_ts.tz_convert("Asia/Kolkata")

        target_time = current_ts - pd.Timedelta(hours=target_hours)
        tolerance = max(30 * 60, int(target_hours * 60 * 60 * 0.5))
        best_value = None
        best_delta = None

        with self._m3_history_lock:
            for timestamp, value in self._m3_history:
                ts = pd.Timestamp(timestamp)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("Asia/Kolkata")
                else:
                    ts = ts.tz_convert("Asia/Kolkata")

                if ts > current_ts:
                    continue
                delta = abs((ts - target_time).total_seconds())
                if delta > tolerance:
                    continue
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_value = float(value)

        return float(best_value) if best_value is not None else None

    def prepare_m3_input(
        self,
        esp32_data: Dict[str, Any],
        weather_data: Dict[str, Any],
        current_datetime: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if current_datetime is None:
            current_datetime = datetime.now().astimezone(ZoneInfo("Asia/Kolkata"))
        else:
            current_ts = pd.Timestamp(current_datetime)
            if current_ts.tzinfo is None:
                current_ts = current_ts.tz_localize("Asia/Kolkata")
            else:
                current_ts = current_ts.tz_convert("Asia/Kolkata")
            current_datetime = current_ts.to_pydatetime()

        soil_moisture = esp32_data.get("soil_moisture_pct", esp32_data.get("soil_moisture_percent"))
        if soil_moisture is None and esp32_data.get("soil_moisture_raw") is not None:
            soil_moisture = self.soil_raw_to_percent(float(esp32_data["soil_moisture_raw"]))
        if soil_moisture is None:
            raise FeatureError("M3 requires soil_moisture_pct or soil_moisture_raw.")

        air_temp = esp32_data.get("air_temperature_c", esp32_data.get("temperature"))
        air_humidity = esp32_data.get("air_humidity_pct", esp32_data.get("humidity"))
        soil_temp = esp32_data.get("soil_temperature_c", esp32_data.get("soil_temperature"))
        if air_temp is None or air_humidity is None or soil_temp is None:
            raise FeatureError("M3 requires air_temperature_c, air_humidity_pct, and soil_temperature_c.")

        moisture_value = self._require_finite_float(soil_moisture, "M3 soil_moisture_pct")
        air_temp_val = self._require_finite_float(air_temp, "M3 air_temperature_c")
        air_humidity_val = self._require_finite_float(air_humidity, "M3 air_humidity_pct")
        soil_temp_val = self._require_finite_float(soil_temp, "M3 soil_temperature_c")
        if not 0.0 <= moisture_value <= 100.0:
            raise FeatureError("M3 soil_moisture_pct must be between 0 and 100.")
        if not 0.0 <= air_humidity_val <= 100.0:
            raise FeatureError("M3 air_humidity_pct must be between 0 and 100.")

        self.record_m3_reading(esp32_data, current_datetime)
        moisture_lag_1h = self._nearest_past_value(current_datetime, 1.0)
        moisture_lag_3h = self._nearest_past_value(current_datetime, 3.0)
        fallback_lags = []
        if moisture_lag_1h is None:
            moisture_lag_1h = moisture_value
            fallback_lags.append("moisture_lag_1h")
        if moisture_lag_3h is None:
            moisture_lag_3h = moisture_value
            fallback_lags.append("moisture_lag_3h")

        rain_now_mm = weather_data.get("rain_now_mm", weather_data.get("rainfall_mm", weather_data.get("precipitation_mm")))
        if rain_now_mm is None:
            rain_now_mm = 0.0
        rain_now_mm = self._require_finite_float(rain_now_mm, "M3 rain_now_mm")
        if rain_now_mm < 0.0:
            raise FeatureError("M3 rain_now_mm cannot be negative.")

        region_name = (
            str(weather_data.get("site") or weather_data.get("location") or weather_data.get("region") or "Chikkamagaluru").strip()
            or PRIMARY_REGION
        )
        rain_sum_6h_mm = weather_data.get("rain_sum_6h_mm")
        if rain_sum_6h_mm is None:
            rain_sum_6h_mm = get_m3_rain_sum_6h(region=region_name, current_datetime=current_datetime)
        rain_sum_6h_mm = self._require_finite_float(rain_sum_6h_mm, "M3 rain_sum_6h_mm")
        if rain_sum_6h_mm < 0.0:
            raise FeatureError("M3 rain_sum_6h_mm cannot be negative.")

        self._validate_required(
            {
                "site": esp32_data.get("site") or weather_data.get("site") or weather_data.get("location") or "default_site",
                "air_temperature_c": air_temp_val,
                "air_humidity_pct": air_humidity_val,
                "soil_temperature_c": soil_temp_val,
                "soil_moisture_pct": moisture_value,
                "moisture_lag_1h": moisture_lag_1h,
                "moisture_lag_3h": moisture_lag_3h,
                "rain_now_mm": rain_now_mm,
                "rain_sum_6h_mm": rain_sum_6h_mm,
            },
            [
                "site",
                "air_temperature_c",
                "air_humidity_pct",
                "soil_temperature_c",
                "soil_moisture_pct",
                "moisture_lag_1h",
                "moisture_lag_3h",
                "rain_now_mm",
                "rain_sum_6h_mm",
            ],
            "M3",
        )

        hour = current_datetime.hour
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)

        site = str(esp32_data.get("site") or weather_data.get("site") or weather_data.get("location") or "default_site")
        data = {
            "site": site,
            "air_temperature_c": air_temp_val,
            "air_humidity_pct": air_humidity_val,
            "soil_temperature_c": soil_temp_val,
            "soil_moisture_pct": moisture_value,
            "moisture_lag_1h": float(moisture_lag_1h),
            "moisture_lag_3h": float(moisture_lag_3h),
            "rain_now_mm": float(rain_now_mm),
            "rain_sum_6h_mm": float(rain_sum_6h_mm),
            "hour_sin": float(hour_sin),
            "hour_cos": float(hour_cos),
        }
        self._validate_required(data, M3_EXPECTED_FEATURES, "M3")
        frame = pd.DataFrame([data], columns=M3_EXPECTED_FEATURES)
        frame.attrs["history_fallback_lags"] = fallback_lags
        return frame

    def predict_m3(
        self,
        esp32_data: Dict[str, Any],
        weather_data: Dict[str, Any],
        current_datetime: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        try:
            frame = self.prepare_m3_input(esp32_data, weather_data, current_datetime=current_datetime)
            model = self._require_loaded_model(self.m3_model, "M3")
            prediction = self._predict_with_pipeline(model, frame)
            value = float(prediction[0])
            clamped = max(0.0, min(100.0, value))
            fallback_lags = frame.attrs.get("history_fallback_lags", [])
            result = {
                "success": True,
                "module": "M3_Future_Soil_Moisture",
                "model": "LightGBM",
                "target_moisture_1h": clamped,
                "future_soil_moisture_percent": clamped,
                "history_fallback": bool(fallback_lags),
                "history_fallback_lags": fallback_lags,
                "features": frame.iloc[0].to_dict(),
            }
            if fallback_lags:
                logger.warning("[M3] Using current moisture for missing lag inputs: %s", fallback_lags)
            logger.info("[M3] Soil moisture after 1h: %.2f%%", clamped)
            return result
        except HistoryUnavailableError as exc:
            logger.warning("[M3] prediction unavailable: %s", exc)
            return {
                "success": False,
                "module": "M3_Future_Soil_Moisture",
                "error_code": "insufficient_history",
                "error": str(exc),
            }
        except FeatureError as exc:
            logger.warning("[M3] invalid prediction input: %s", exc)
            return {
                "success": False,
                "module": "M3_Future_Soil_Moisture",
                "error_code": "invalid_features",
                "error": str(exc),
            }
        except Exception as exc:
            logger.exception("[M3] prediction failed: %s", exc)
            return {"success": False, "module": "M3_Future_Soil_Moisture", "error": str(exc)}

    @staticmethod
    def build_m4_row(region: str, weather_data: Dict[str, Any], current_datetime: Optional[datetime] = None) -> Dict[str, Any]:
        if current_datetime is None:
            current_datetime = datetime.now()
        return {
            "district": str(region),
            "year": int(current_datetime.year),
            "month": int(current_datetime.month),
            "day_of_year": int(current_datetime.timetuple().tm_yday),
            "hour": int(current_datetime.hour),
            "rainfall_mm": float(weather_data.get("rainfall_mm", weather_data.get("rain_now_mm", 0.0))),
            "temperature_c": float(weather_data.get("temperature_c", 0.0)),
            "humidity_percent": float(weather_data.get("humidity_percent", 0.0)),
            "pressure_msl_hpa": float(weather_data.get("pressure_msl_hpa", weather_data.get("surface_pressure_hpa", 0.0))),
            "cloud_cover_percent": float(weather_data.get("cloud_cover_percent", 0.0)),
            "wind_speed_kmh": float(weather_data.get("wind_speed_kmh", 0.0)),
        }

    def prepare_m4_input(self, weather_row: Dict[str, Any]) -> pd.DataFrame:
        self._validate_required(weather_row, M4_EXPECTED_FEATURES, "M4")
        return pd.DataFrame([weather_row], columns=M4_EXPECTED_FEATURES)

    def predict_m4(self, weather_row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            frame = self.prepare_m4_input(weather_row)
            model = self._require_loaded_model(self.m4_model, "M4")
            prediction = self._predict_with_pipeline(model, frame)
            value = float(prediction[0])
            result = {
                "success": True,
                "module": "M4_Hourly_Rainfall",
                "model": "XGBoost",
                "future_rainfall_mm": value,
                "predicted_rainfall_mm": value,
                "features": frame.iloc[0].to_dict(),
            }
            logger.info("[M4] Rain forecast next hour: %.2f mm", value)
            return result
        except Exception as exc:
            logger.exception("[M4] prediction failed: %s", exc)
            return {"success": False, "module": "M4_Hourly_Rainfall", "error": str(exc)}

    def predict_m4_latest(self, region: str = PRIMARY_REGION) -> Dict[str, Any]:
        try:
            weather_data = get_current_weather_for_region(region)
        except Exception as exc:
            return {"success": False, "module": "M4_Hourly_Rainfall", "error": f"Weather unavailable: {exc}"}
        normalized_region = str(weather_data.get("location") or region)
        row = self.build_m4_row(region=normalized_region, weather_data=weather_data, current_datetime=datetime.now())
        return self.predict_m4(row)

    def run_live_predictions(
        self,
        esp32_data: Dict[str, Any],
        location: str,
        soil_type: str,
        irrigation: Optional[str] = None,
        crop_type: Optional[str] = None,
        best_time: str = "Any",
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        reference_date = datetime.now()
        normalized_region = str(region or location or PRIMARY_REGION).strip() or PRIMARY_REGION

        soil_moisture = esp32_data.get("soil_moisture_percent")
        if soil_moisture is None:
            soil_moisture = esp32_data.get("soil_moisture_pct")
        if soil_moisture is None and esp32_data.get("soil_moisture_raw") is not None:
            soil_moisture = self.soil_raw_to_percent(float(esp32_data["soil_moisture_raw"]))

        weather = get_current_weather_for_location(normalized_region) if normalized_region in {"Bangalore", "Chikkamagaluru", "Chikmangaluru", "Davangere", "Gulbarga", "Hassan", "Kasaragodu", "Kodagu", "Madikeri", "Mangalore", "Mysuru", "Raichur"} else {"temperature_c": 0.0, "humidity_percent": 0.0, "rainfall_mm": 0.0, "location": normalized_region}
        result = {
            "m1": self.predict_m1(
                location=location,
                soil_type=soil_type,
                irrigation=irrigation,
                temperature=float(weather.get("temperature_c", 0.0)),
                humidity=float(weather.get("humidity_percent", 0.0)),
                soil_moisture=float(soil_moisture) if soil_moisture is not None else 0.0,
                reference_date=reference_date,
                region=normalized_region,
            ),
            "m2": self.predict_m2(
                esp32_data=esp32_data,
                crop_type=crop_type,
                best_time=best_time,
            ),
            "m3": self.predict_m3(
                esp32_data=esp32_data,
                weather_data={**weather, "rain_sum_6h_mm": get_m3_rain_sum_6h(region=normalized_region, current_datetime=reference_date)},
                current_datetime=reference_date,
            ),
            "m4": self.predict_m4_latest(region=normalized_region),
        }
        return result

    def model_status(self) -> Dict[str, Any]:
        m4_dir = self.model_dir / "M4_Rainfall_Forecasting"
        if not m4_dir.exists():
            m4_dir = self.model_dir / "M4_Hourly_Rainfall"
        return {
            "M1": {"loaded": self.m1_model is not None, "model": "Random Forest", "file": str(_resolve_model_file(self.model_dir / "M1_Crop_Recommendation", "random_forest.joblib"))},
            "M2": {"loaded": self.m2_model is not None, "model": "LightGBM", "file": str(_resolve_model_file(self.model_dir / "M2_Context_Aware_Irrigation", "lightgbm.joblib"))},
            "M3": {"loaded": self.m3_model is not None, "model": "LightGBM", "file": str(_resolve_model_file(self.model_dir / "M3_Future_Soil_Moisture", "lightgbm.joblib"))},
            "M4": {"loaded": self.m4_model is not None, "model": "XGBoost", "file": str(_resolve_model_file(m4_dir, "xgboost.joblib"))},
        }


_ml_engine: Optional[MLEngine] = None
_ml_engine_lock = threading.Lock()


def get_ml_engine() -> MLEngine:
    global _ml_engine
    if _ml_engine is None:
        with _ml_engine_lock:
            if _ml_engine is None:
                _ml_engine = MLEngine()
    return _ml_engine


def predict_m1(
    location: str,
    soil_type: str,
    irrigation: Optional[str] = None,
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    reference_date: Optional[datetime] = None,
    region: str = PRIMARY_REGION,
    annual_rainfall_mm: Optional[float] = None,
    soil_moisture: Optional[float] = None,
    year: Optional[int] = None,
) -> Dict[str, Any]:
    return get_ml_engine().predict_m1(
        location=location,
        soil_type=soil_type,
        irrigation=irrigation,
        temperature=temperature,
        humidity=humidity,
        reference_date=reference_date,
        region=region,
        annual_rainfall_mm=annual_rainfall_mm,
        soil_moisture=soil_moisture,
        year=year,
    )


def predict_m2(
    esp32_data: Dict[str, Any],
    crop_type: Optional[str] = None,
    best_time: str = "Any",
    current_datetime: Optional[datetime] = None,
) -> Dict[str, Any]:
    return get_ml_engine().predict_m2(
        esp32_data=esp32_data,
        crop_type=crop_type,
        best_time=best_time,
        current_datetime=current_datetime,
    )


def predict_m3(
    esp32_data: Dict[str, Any],
    weather_data: Dict[str, Any],
    current_datetime: Optional[datetime] = None,
) -> Dict[str, Any]:
    return get_ml_engine().predict_m3(
        esp32_data=esp32_data,
        weather_data=weather_data,
        current_datetime=current_datetime,
    )


def predict_m4(weather_row: Dict[str, Any]) -> Dict[str, Any]:
    return get_ml_engine().predict_m4(weather_row)
