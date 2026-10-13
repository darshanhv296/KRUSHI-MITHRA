"""
Krushi Mithra
Smart Farming & Intelligent Irrigation System

weather.py
==========
Weather/data layer for the Krushi Mithra runtime.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (
    DATA_DIR,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    PROJECT_ROOT,
    WEATHER_LATITUDE,
    WEATHER_LONGITUDE,
    WEATHER_TIMEZONE,
)

logger = logging.getLogger("KrushiMithra.Weather")

M1_YEARLY_FILE = DATA_DIR / "M1_yearly_weather.csv"
M4_HOURLY_FILE = DATA_DIR / "M4_hourly_weather.csv"
CROP_RECOMMENDATION_DATASET = PROJECT_ROOT / "DATASET" / "Crop_Recommendation.csv"

PRIMARY_REGION = "Chikkamagaluru"
KOLKATA_TZ = "Asia/Kolkata"

LOCATION_ALIASES = {
    "chikmangaluru": "Chikkamagaluru",
    "chikkamagaluru": "Chikkamagaluru",
    "chikmagalur": "Chikkamagaluru",
    "chikkamagalur": "Chikkamagaluru",
    "bangalore": "Bengaluru Urban",
    "bengaluru": "Bengaluru Urban",
    "davangere": "Davanagere",
    "gulbarga": "Kalaburagi",
    "hassan": "Hassan",
    "kodagu": "Kodagu",
    "madikeri": "Kodagu",
    "mangalore": "Dakshina Kannada",
    "mysuru": "Mysuru",
    "raichur": "Raichur",
    "karwar": "Uttara Kannada",
}

def normalize_location_name(location: Any) -> str:
    if location is None:
        raise ValueError("Location is required.")

    raw = str(location).strip()
    if not raw:
        raise ValueError("Location is required.")

    key = " ".join(raw.split()).casefold()
    if key in LOCATION_ALIASES:
        return LOCATION_ALIASES[key]

    for candidate in sorted(M1_LOCATION_COORDINATES):
        if candidate.casefold() == key:
            return candidate

    raise ValueError(f"Unsupported M1 location: {raw}.")


def normalize_datetime(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp("NaT")

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return pd.Timestamp("NaT")

    if ts.tzinfo is None:
        return ts.tz_localize(KOLKATA_TZ)
    return ts.tz_convert(KOLKATA_TZ)


def normalize_datetime_series(series: Any) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if s.empty:
        return s

    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize(KOLKATA_TZ)
    return s.dt.tz_convert(KOLKATA_TZ)


def normalize_weather_region(region: Any) -> str:
    if region is None:
        raise ValueError("Region is required.")

    raw = " ".join(str(region).split())
    if not raw:
        raise ValueError("Region is required.")

    key = raw.casefold()
    canonical = M4_REGION_ALIASES.get(key, raw)
    for candidate in M4_CURRENT_REGION_COORDINATES:
        if candidate.casefold() == canonical.casefold():
            return candidate

    raise ValueError(f"Unsupported weather region: {raw}.")


# Initial historical start date following trained dataset (2025-01-01 -> 2025-12-31)
INITIAL_START_DATE = "2026-01-01"

REGIONS = {
    "Bagalkot": (16.1691, 75.6615),
    "Ballari": (15.1394, 76.9214),
    "Belagavi": (15.8497, 74.4977),
    "Bengaluru Rural": (13.2929, 77.7512),
    "Bengaluru Urban": (12.9716, 77.5946),
    "Bidar": (17.9104, 77.5199),
    "Chamarajanagar": (11.9261, 76.9437),
    "Chikkaballapura": (13.4355, 77.7315),
    "Chikkamagaluru": (13.3161, 75.7720),
    "Chitradurga": (14.2251, 76.3980),
    "Dakshina Kannada": (12.9141, 74.8560),
    "Davanagere": (14.4644, 75.9218),
    "Dharwad": (15.4589, 75.0078),
    "Gadag": (15.4315, 75.6355),
    "Hassan": (13.0033, 76.1004),
    "Haveri": (14.7951, 75.3991),
    "Kalaburagi": (17.3297, 76.8343),
    "Kodagu": (12.4244, 75.7382),
    "Kolar": (13.1367, 78.1292),
    "Koppal": (15.3505, 76.1548),
    "Mandya": (12.5218, 76.8951),
    "Mysuru": (12.2958, 76.6394),
    "Raichur": (16.2120, 77.3439),
    "Ramanagara": (12.7150, 77.2810),
    "Shivamogga": (13.9299, 75.5681),
    "Tumakuru": (13.3392, 77.1130),
    "Udupi": (13.3409, 74.7421),
    "Uttara Kannada": (14.8136, 74.1297),
    "Vijayanagara": (15.2689, 76.3909),
    "Vijayapura": (16.8302, 75.7100),
    "Yadgir": (16.7700, 77.1373),
}

M1_LOCATION_COORDINATES = dict(REGIONS)
M1_ALLOWED_LOCATIONS = set(M1_LOCATION_COORDINATES)
M4_CURRENT_REGION_COORDINATES = dict(REGIONS)
M4_REGION_ALIASES = {
    "chikmangaluru": "Chikkamagaluru",
    "chikmagalur": "Chikkamagaluru",
    "chikkamagalur": "Chikkamagaluru",
    "gulbarga": "Kalaburagi",
    "bangalore": "Bengaluru Urban",
    "bengaluru": "Bengaluru Urban",
    "davangere": "Davanagere",
    "mangalore": "Dakshina Kannada",
    "madikeri": "Kodagu",
    "karwar": "Uttara Kannada",
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
    "rain",
    "precipitation",
]

M4_MODEL_FEATURES = [
    "temperature_c",
    "humidity_percent",
    "dew_point_c",
    "pressure_msl_hpa",
    "surface_pressure_hpa",
    "cloud_cover_percent",
    "wind_speed_kmh",
    "wind_direction_deg",
    "wind_gusts_kmh",
    "vapour_pressure_deficit_kpa",
    "rainfall_mm",
    "precipitation_mm",
    "hour",
    "day_of_year",
    "month",
]

M4_COLUMNS = [
    "region",
    "latitude",
    "longitude",
    "date",
    "time",
    "temperature_c",
    "humidity_percent",
    "dew_point_c",
    "pressure_msl_hpa",
    "surface_pressure_hpa",
    "cloud_cover_percent",
    "wind_speed_kmh",
    "wind_direction_deg",
    "wind_gusts_kmh",
    "vapour_pressure_deficit_kpa",
    "rainfall_mm",
    "precipitation_mm",
    "hour",
    "day_of_year",
    "month",
    "future_rainfall_mm",
]

_M4_SYNC_LOCK = threading.Lock()
_M4_SYNC_RUNNING = False

M1_COLUMNS = [
    "region",
    "date",
    "year",
    "temperature_c",
    "humidity_percent",
    "rainfall_mm",
]


class WeatherError(Exception):
    """Base weather module exception."""


class WeatherAPIError(WeatherError):
    """Raised when Open-Meteo returns an error."""


class WeatherDataError(WeatherError):
    """Raised when weather data is invalid or incomplete."""


class WeatherClient:
    """Open-Meteo API client."""

    def __init__(
        self,
        latitude: float = WEATHER_LATITUDE,
        longitude: float = WEATHER_LONGITUDE,
        timezone: str = WEATHER_TIMEZONE,
        timeout: int = 30,
    ):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.timezone = timezone
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Krushi-Mithra/1.0"})

    def close(self) -> None:
        """Closes the underlying requests session."""
        if self.session:
            self.session.close()

    def __enter__(self) -> WeatherClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WeatherAPIError(f"Weather API request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise WeatherAPIError("Weather API returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise WeatherAPIError("Weather API returned an invalid response.")

        if data.get("error"):
            raise WeatherAPIError(data.get("reason", "Unknown Open-Meteo error."))

        return data

    def get_current_weather(self) -> Dict[str, Any]:
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": ",".join((*HOURLY_VARIABLES, "weather_code")),
            "hourly": ",".join((
                "precipitation_probability",
                "precipitation",
                "rain",
                "weather_code",
            )),
            "timezone": self.timezone,
        }
        data = self._get(OPEN_METEO_FORECAST_URL, params)
        current = data.get("current")

        if not current:
            raise WeatherDataError("Current weather data is missing.")

        result = {
            "timestamp": current.get("time"),
            "temperature_c": float(current.get("temperature_2m") or 0.0),
            "humidity_percent": float(current.get("relative_humidity_2m") or 0.0),
            "dew_point_c": float(current.get("dew_point_2m") or 0.0),
            "pressure_msl_hpa": float(current.get("pressure_msl") or 0.0),
            "surface_pressure_hpa": float(current.get("surface_pressure") or 0.0),
            "cloud_cover_percent": float(current.get("cloud_cover") or 0.0),
            "wind_speed_kmh": float(current.get("wind_speed_10m") or 0.0),
            "wind_direction_deg": float(current.get("wind_direction_10m") or 0.0),
            "wind_gusts_kmh": float(current.get("wind_gusts_10m") or 0.0),
            "vapour_pressure_deficit_kpa": float(current.get("vapour_pressure_deficit") or 0.0),
            "rainfall_mm": float(current.get("rain") or 0.0),
            "precipitation_mm": float(current.get("precipitation") or 0.0),
            "weather_code": current.get("weather_code"),
        }
        hourly = data.get("hourly") or {}
        hourly_times = hourly.get("time") or []
        if hourly_times:
            current_time = str(current.get("time") or "")
            next_hour_index = next(
                (
                    index
                    for index, forecast_time in enumerate(hourly_times)
                    if str(forecast_time) > current_time
                ),
                0,
            )
            result["next_hour_timestamp"] = hourly_times[next_hour_index]
            for source_name, result_name in (
                ("precipitation_probability", "next_hour_rain_probability_percent"),
                ("precipitation", "next_hour_precipitation_mm"),
                ("rain", "next_hour_rainfall_mm"),
                ("weather_code", "next_hour_weather_code"),
            ):
                values = hourly.get(source_name) or []
                if next_hour_index < len(values):
                    result[result_name] = values[next_hour_index]
        return result

    def fetch_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        region: str,
    ) -> pd.DataFrame:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": self.timezone,
        }
        logger.info("Fetching %s: %s -> %s", region, start_date, end_date)
        data = self._get(OPEN_METEO_ARCHIVE_URL, params)
        hourly = data.get("hourly")

        if not hourly:
            raise WeatherDataError(f"No hourly weather returned for {region}.")

        return self._convert_hourly_response(
            hourly=hourly,
            latitude=latitude,
            longitude=longitude,
            region=region,
        )

    @staticmethod
    def _convert_hourly_response(
        hourly: Dict[str, Any],
        latitude: float,
        longitude: float,
        region: str,
    ) -> pd.DataFrame:
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        def as_series(values: Any) -> pd.Series:
            return pd.Series(pd.to_numeric(values, errors="coerce"), dtype="float64").fillna(0.0)

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(times, errors="coerce"),
                "temperature_c": as_series(hourly.get("temperature_2m")),
                "humidity_percent": as_series(hourly.get("relative_humidity_2m")),
                "dew_point_c": as_series(hourly.get("dew_point_2m")),
                "pressure_msl_hpa": as_series(hourly.get("pressure_msl")),
                "surface_pressure_hpa": as_series(hourly.get("surface_pressure")),
                "cloud_cover_percent": as_series(hourly.get("cloud_cover")),
                "wind_speed_kmh": as_series(hourly.get("wind_speed_10m")),
                "wind_direction_deg": as_series(hourly.get("wind_direction_10m")),
                "wind_gusts_kmh": as_series(hourly.get("wind_gusts_10m")),
                "vapour_pressure_deficit_kpa": as_series(hourly.get("vapour_pressure_deficit")),
                "rainfall_mm": as_series(hourly.get("rain")),
                "precipitation_mm": as_series(hourly.get("precipitation")),
            }
        )
        df = df.dropna(subset=["datetime"])
        df["region"] = region
        df["latitude"] = latitude
        df["longitude"] = longitude
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        df["time"] = df["datetime"].dt.strftime("%H:%M")
        df["hour"] = df["datetime"].dt.hour
        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["month"] = df["datetime"].dt.month
        df["future_rainfall_mm"] = None
        return df


class M1WeatherHistory:
    """Maintains the M1 daily weather dataset."""

    def __init__(self, csv_path: Path = M1_YEARLY_FILE):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=M1_COLUMNS)

        df = pd.read_csv(self.csv_path)
        if df.empty:
            return pd.DataFrame(columns=M1_COLUMNS)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(0.0)
        df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
        df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
        df = df.dropna(subset=["date"])
        return df

    def append_daily_data(self, hourly_df: pd.DataFrame) -> int:
        if hourly_df.empty:
            return 0

        df = hourly_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["date"] = df["datetime"].dt.normalize()

        daily = (
            df.groupby(["region", "date"], as_index=False)
            .agg(
                temperature_c=("temperature_c", "mean"),
                humidity_percent=("humidity_percent", "mean"),
                rainfall_mm=("rainfall_mm", "sum"),
            )
        )

        existing = self.load()
        if not existing.empty:
            existing_keys = set(
                zip(
                    existing["region"].astype(str),
                    existing["date"].dt.normalize(),
                )
            )
            daily = daily[
                ~daily.apply(
                    lambda row: (str(row["region"]), row["date"]) in existing_keys,
                    axis=1,
                )
            ].copy()

        if daily.empty:
            logger.info("M1 weather: no new daily observations.")
            return 0

        daily["year"] = daily["date"].dt.year

        combined = pd.concat([existing, daily], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["region", "date"], keep="first")
        combined = combined.sort_values(["region", "date"]).reset_index(drop=True)

        save_df = combined[M1_COLUMNS].copy()
        save_df["date"] = pd.to_datetime(save_df["date"]).dt.strftime("%Y-%m-%d")
        save_df.to_csv(self.csv_path, index=False)

        logger.info("M1 weather: appended %d daily rows.", len(daily))
        return len(daily)

    def rolling_365d_rainfall(
        self,
        region: str = PRIMARY_REGION,
        reference_time: Optional[datetime] = None,
    ) -> float:
        df = self.load()
        if df.empty:
            return 0.0

        df = df[df["region"] == region].copy()
        if df.empty:
            return 0.0

        if reference_time is None:
            reference = df["date"].max()
        else:
            reference = pd.Timestamp(reference_time)
            if reference.tzinfo is not None:
                reference = reference.tz_convert(ZoneInfo(KOLKATA_TZ)).tz_localize(None)
            else:
                reference = reference.tz_localize(ZoneInfo(KOLKATA_TZ)).tz_localize(None)
            reference = reference.normalize()

        start = reference - pd.Timedelta(days=365)
        window = df[(df["date"] > start) & (df["date"] <= reference)]
        return float(window["rainfall_mm"].sum())

    def yearly_summary(self, region: str = PRIMARY_REGION) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return pd.DataFrame()

        df = df[df["region"] == region].copy()
        if df.empty:
            return pd.DataFrame()

        df["year"] = df["date"].dt.year
        return (
            df.groupby(["region", "year"], as_index=False)
            .agg(
                temperature_c=("temperature_c", "mean"),
                humidity_percent=("humidity_percent", "mean"),
                rainfall_mm=("rainfall_mm", "sum"),
            )
        )


class M4WeatherHistory:
    """Maintains the M4 hourly weather dataset."""

    def __init__(self, csv_path: Path = M4_HOURLY_FILE):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame(columns=M4_COLUMNS + ["datetime"])

        df = pd.read_csv(self.csv_path)
        if df.empty:
            return pd.DataFrame(columns=M4_COLUMNS + ["datetime"])

        df["datetime"] = normalize_datetime_series(
            df["date"].astype(str) + " " + df["time"].astype(str)
        )

        numeric_columns = [
            "latitude",
            "longitude",
            "temperature_c",
            "humidity_percent",
            "dew_point_c",
            "pressure_msl_hpa",
            "surface_pressure_hpa",
            "cloud_cover_percent",
            "wind_speed_kmh",
            "wind_direction_deg",
            "wind_gusts_kmh",
            "vapour_pressure_deficit_kpa",
            "rainfall_mm",
            "precipitation_mm",
            "hour",
            "day_of_year",
            "month",
            "future_rainfall_mm",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=["region", "datetime"])

    def latest_timestamp(self, region: str) -> Optional[pd.Timestamp]:
        df = self.load()
        if df.empty:
            return None
        region_df = df[df["region"] == region]
        return None if region_df.empty else region_df["datetime"].max()

    def append_new_data(self, new_df: pd.DataFrame) -> int:
        if new_df.empty:
            return 0

        incoming = new_df.copy()
        incoming["datetime"] = normalize_datetime_series(incoming["datetime"])
        incoming = incoming.dropna(subset=["region", "datetime"])
        incoming = incoming.drop_duplicates(subset=["region", "datetime"], keep="last")

        existing = self.load()
        if not existing.empty:
            existing_keys = set(
                zip(
                    existing["region"].astype(str),
                    existing["datetime"],
                )
            )
            incoming = incoming[
                ~incoming.apply(
                    lambda row: (str(row["region"]), row["datetime"]) in existing_keys,
                    axis=1,
                )
            ].copy()

        if incoming.empty:
            logger.info("M4 weather: no new hourly observations.")
            return 0

        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(subset=["region", "datetime"], keep="first")
        combined = combined.sort_values(["region", "datetime"]).reset_index(drop=True)

        next_dt = combined.groupby("region")["datetime"].shift(-1)
        next_rainfall = combined.groupby("region")["rainfall_mm"].shift(-1)
        is_consecutive_hour = (next_dt - combined["datetime"]) == pd.Timedelta(hours=1)
        combined["future_rainfall_mm"] = next_rainfall.where(is_consecutive_hour, None)

        combined["date"] = combined["datetime"].dt.strftime("%Y-%m-%d")
        combined["time"] = combined["datetime"].dt.strftime("%H:%M")
        combined["hour"] = combined["datetime"].dt.hour
        combined["day_of_year"] = combined["datetime"].dt.dayofyear
        combined["month"] = combined["datetime"].dt.month

        save_df = combined[M4_COLUMNS].copy()
        save_df.to_csv(self.csv_path, index=False)

        logger.info("M4 weather: appended %d hourly rows.", len(incoming))
        return len(incoming)

    def sync_region(
        self,
        client: WeatherClient,
        region: str,
        latitude: float,
        longitude: float,
        start_date_if_empty: str = INITIAL_START_DATE,
    ) -> int:
        latest = self.latest_timestamp(region)
        now = datetime.now()
        completed_hour = now.replace(minute=0, second=0, microsecond=0)

        if latest is None:
            start_datetime = normalize_datetime(start_date_if_empty)
        else:
            start_datetime = normalize_datetime(latest + pd.Timedelta(hours=1))

        completed_ts = normalize_datetime(completed_hour)
        if start_datetime > completed_ts:
            logger.info("M4 %s: already up to date.", region)
            return 0

        start_date = start_datetime.date().isoformat()
        end_date = completed_hour.date().isoformat()

        logger.info("M4 %s synchronization: %s -> %s", region, start_datetime, completed_hour)

        new_df = client.fetch_historical(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            region=region,
        )
        if new_df.empty:
            return 0

        new_df["datetime"] = normalize_datetime_series(new_df["datetime"])
        new_df = new_df[
            (new_df["datetime"] >= start_datetime)
            & (new_df["datetime"] <= completed_ts)
        ].copy()

        if new_df.empty:
            return 0

        return self.append_new_data(new_df)

    def sync_all_regions(self, start_date_if_empty: str = INITIAL_START_DATE) -> int:
        client = WeatherClient()
        total_added = 0
        try:
            for region, (latitude, longitude) in REGIONS.items():
                try:
                    added = self.sync_region(
                        client=client,
                        region=region,
                        latitude=latitude,
                        longitude=longitude,
                        start_date_if_empty=start_date_if_empty,
                    )
                    total_added += added
                except Exception as exc:
                    logger.error("M4 sync failed for %s: %s", region, exc)

            logger.info("M4 synchronization completed. Total new rows: %d", total_added)
            return total_added
        finally:
            client.close()


def get_current_weather_for_region(region: str) -> Dict[str, Any]:
    normalized = normalize_weather_region(region)
    latitude, longitude = M4_CURRENT_REGION_COORDINATES[normalized]
    now = datetime.now().timestamp()

    with _M1_WEATHER_CACHE_LOCK:
        cached = _M1_WEATHER_CACHE.get(normalized)
        if cached and (now - cached["timestamp"]) < _M1_WEATHER_CACHE_TTL_SECONDS:
            return cached["data"]

    client = WeatherClient(latitude=latitude, longitude=longitude, timezone=WEATHER_TIMEZONE)
    try:
        weather = client.get_current_weather()
        if not weather:
            raise WeatherDataError(f"Weather data unavailable for region: {normalized}.")

        weather["location"] = normalized
        weather["latitude"] = float(latitude)
        weather["longitude"] = float(longitude)
        weather["weather_source"] = "Open-Meteo"

        with _M1_WEATHER_CACHE_LOCK:
            _M1_WEATHER_CACHE[normalized] = {
                "timestamp": now,
                "data": weather,
            }

        return weather
    except WeatherAPIError as exc:
        raise WeatherAPIError(
            f"Weather data unavailable for region: {normalized}."
        ) from exc
    finally:
        client.close()


class M3WeatherNormalizer:
    """Utility helpers to keep M3 weather input aligned with the trained schema."""

    @staticmethod
    def six_hour_rain_sum(
        region: str = PRIMARY_REGION,
        current_datetime: Optional[datetime] = None,
    ) -> float:
        if current_datetime is None:
            current_datetime = datetime.now()

        history = M4WeatherHistory()
        frame = history.load()
        if frame.empty:
            return 0.0

        canonical_region = normalize_weather_region(region)
        region_df = frame[frame["region"].astype(str) == canonical_region].copy()
        if region_df.empty:
            return 0.0

        region_df["datetime"] = normalize_datetime_series(region_df["datetime"])
        region_df = region_df.dropna(subset=["datetime"]).copy()
        if region_df.empty:
            return 0.0

        current_datetime = normalize_datetime(current_datetime).to_pydatetime()
        current_floor = current_datetime.astimezone(ZoneInfo(KOLKATA_TZ)).replace(minute=0, second=0, microsecond=0)
        end_time = current_floor - timedelta(hours=1)
        start_time = end_time - timedelta(hours=5)

        window = region_df[
            (region_df["datetime"] >= pd.Timestamp(start_time))
            & (region_df["datetime"] <= pd.Timestamp(end_time))
        ].copy()

        if window.empty:
            return 0.0

        if "rainfall_mm" not in window.columns:
            return 0.0

        rainfall = window["rainfall_mm"]
        return float(pd.to_numeric(rainfall, errors="coerce").fillna(0.0).sum())


def get_m3_rain_sum_6h(
    region: str = PRIMARY_REGION,
    current_datetime: Optional[datetime] = None,
) -> float:
    return float(M3WeatherNormalizer.six_hour_rain_sum(region=region, current_datetime=current_datetime))


def sync_m1_weather(start_date_if_empty: str = INITIAL_START_DATE) -> int:
    client = WeatherClient()
    history = M1WeatherHistory()
    try:
        existing = history.load()
        if existing.empty:
            start_date = pd.Timestamp(start_date_if_empty)
        else:
            latest = existing["date"].max()
            start_date = latest + pd.Timedelta(days=1)

        end_date = datetime.now().date() - timedelta(days=1)
        if start_date.date() > end_date:
            logger.info("M1 weather is already up to date.")
            return 0

        latitude, longitude = REGIONS[PRIMARY_REGION]
        hourly_df = client.fetch_historical(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date.date().isoformat(),
            end_date=end_date.isoformat(),
            region=PRIMARY_REGION,
        )
        if hourly_df.empty:
            return 0

        return history.append_daily_data(hourly_df)
    finally:
        client.close()


def _run_m4_sync(start_date_if_empty: str = INITIAL_START_DATE) -> int:
    history = M4WeatherHistory()
    return history.sync_all_regions(start_date_if_empty=start_date_if_empty)


def schedule_m4_sync(start_date_if_empty: str = INITIAL_START_DATE) -> bool:
    global _M4_SYNC_RUNNING
    with _M4_SYNC_LOCK:
        if _M4_SYNC_RUNNING:
            logger.info("M4 sync already in progress; skipping queued sync.")
            return False
        _M4_SYNC_RUNNING = True

    def _worker() -> None:
        try:
            logger.info("Starting background M4 weather synchronization.")
            _run_m4_sync(start_date_if_empty=start_date_if_empty)
        except Exception as exc:
            logger.exception("Background M4 synchronization failed: %s", exc)
        finally:
            global _M4_SYNC_RUNNING
            with _M4_SYNC_LOCK:
                _M4_SYNC_RUNNING = False

    worker = threading.Thread(
        target=_worker,
        name="M4WeatherSyncThread",
        daemon=True,
    )
    worker.start()
    return True


def sync_m4_weather(start_date_if_empty: str = INITIAL_START_DATE) -> int:
    global _M4_SYNC_RUNNING
    with _M4_SYNC_LOCK:
        if _M4_SYNC_RUNNING:
            logger.info("M4 weather sync already running; skipping duplicate request.")
            return 0
        _M4_SYNC_RUNNING = True

    try:
        return _run_m4_sync(start_date_if_empty=start_date_if_empty)
    finally:
        with _M4_SYNC_LOCK:
            _M4_SYNC_RUNNING = False


def ensure_m4_history_ready(start_date_if_empty: str = INITIAL_START_DATE) -> int:
    history = M4WeatherHistory()
    if not history.load().empty:
        logger.info("M4 weather history already exists; skipping initial sync.")
        return 0

    logger.info("M4 weather history is empty. Running required initial synchronization in background.")
    schedule_m4_sync(start_date_if_empty=start_date_if_empty)
    return 0


_M1_WEATHER_CACHE: Dict[str, Dict[str, Any]] = {}
_M1_WEATHER_CACHE_LOCK = threading.Lock()
_M1_WEATHER_CACHE_TTL_SECONDS = 300


def get_current_weather() -> Dict[str, Any]:
    client = WeatherClient()
    try:
        return client.get_current_weather()
    finally:
        client.close()


def get_current_weather_at(
    latitude: float,
    longitude: float,
    timezone: str = WEATHER_TIMEZONE,
) -> Dict[str, Any]:
    """Fetch current Open-Meteo weather for supplied coordinates."""
    client = WeatherClient(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )
    try:
        weather = client.get_current_weather()
        weather.update({
            "latitude": float(latitude),
            "longitude": float(longitude),
            "weather_source": "Open-Meteo",
        })
        return weather
    finally:
        client.close()


def get_current_weather_for_location(location: str) -> Dict[str, Any]:
    normalized = normalize_location_name(location)

    latitude, longitude = M1_LOCATION_COORDINATES[normalized]
    now = datetime.now().timestamp()

    with _M1_WEATHER_CACHE_LOCK:
        cached = _M1_WEATHER_CACHE.get(normalized)
        if cached and (now - cached["timestamp"]) < _M1_WEATHER_CACHE_TTL_SECONDS:
            return cached["data"]

    client = WeatherClient(latitude=latitude, longitude=longitude, timezone=WEATHER_TIMEZONE)
    try:
        weather = client.get_current_weather()
        if not weather:
            raise WeatherDataError(f"Weather data unavailable for selected location: {normalized}.")

        weather["location"] = normalized
        weather["latitude"] = float(latitude)
        weather["longitude"] = float(longitude)
        weather["weather_source"] = "Open-Meteo"

        with _M1_WEATHER_CACHE_LOCK:
            _M1_WEATHER_CACHE[normalized] = {
                "timestamp": now,
                "data": weather,
            }

        return weather
    except WeatherAPIError as exc:
        raise WeatherAPIError(
            f"Weather data unavailable for selected location: {normalized}."
        ) from exc
    finally:
        client.close()


def get_m1_rolling_365d_rainfall(region: str = PRIMARY_REGION) -> float:
    return M1WeatherHistory().rolling_365d_rainfall(region=region)


@lru_cache(maxsize=1)
def _load_crop_rainfall_reference() -> Dict[str, float]:
    if not CROP_RECOMMENDATION_DATASET.exists():
        raise FileNotFoundError(
            f"Crop recommendation dataset not found: {CROP_RECOMMENDATION_DATASET}"
        )

    frame = pd.read_csv(
        CROP_RECOMMENDATION_DATASET,
        usecols=["Location", "Annual_Rainfall_mm"],
    )
    frame["Location"] = frame["Location"].astype(str).str.strip()
    frame["Annual_Rainfall_mm"] = pd.to_numeric(
        frame["Annual_Rainfall_mm"],
        errors="coerce",
    )
    reference = frame.dropna(subset=["Location", "Annual_Rainfall_mm"])
    return {
        str(location): float(rainfall)
        for location, rainfall in reference.groupby("Location")["Annual_Rainfall_mm"].median().items()
    }


def get_m1_annual_rainfall_reference(location: str) -> float:
    normalized = normalize_location_name(location)
    rainfall_by_location = _load_crop_rainfall_reference()
    try:
        return rainfall_by_location[normalized]
    except KeyError as exc:
        raise WeatherDataError(
            f"No annual rainfall reference is available for {normalized}."
        ) from exc


def get_m1_yearly_summary(region: str = PRIMARY_REGION) -> pd.DataFrame:
    return M1WeatherHistory().yearly_summary(region=region)