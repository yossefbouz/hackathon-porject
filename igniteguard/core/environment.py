"""External data acquisition helpers for wind and land-cover features."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
MODIS_LAND_COVER_ENDPOINT = (
    "https://modis.ornl.gov/rst/api/v1/MCD12Q1/LC_Type1/subset"
)

USER_AGENT = "IgniteGuard/0.1 (+https://example.com/igniteguard)"


@dataclass(frozen=True)
class Location:
    """Simple geographic coordinate container."""

    identifier: str
    latitude: float
    longitude: float


IGBP_CLASSES = {
    0: "Water",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren or Sparsely Vegetated",
    254: "Unclassified",
    255: "Fill Value",
}

FLAMMABILITY_SCORES = {
    "Water": 0.0,
    "Permanent Snow and Ice": 0.0,
    "Barren or Sparsely Vegetated": 0.2,
    "Permanent Wetlands": 0.3,
    "Croplands": 0.4,
    "Cropland/Natural Vegetation Mosaics": 0.5,
    "Evergreen Broadleaf Forest": 0.6,
    "Deciduous Broadleaf Forest": 0.6,
    "Mixed Forests": 0.65,
    "Evergreen Needleleaf Forest": 0.7,
    "Deciduous Needleleaf Forest": 0.7,
    "Woody Savannas": 0.7,
    "Closed Shrublands": 0.75,
    "Savannas": 0.75,
    "Open Shrublands": 0.8,
    "Grasslands": 0.9,
    "Urban and Built-up Lands": 0.1,
}


def _circular_mean(values: Sequence[float]) -> float:
    """Return the mean direction (degrees) accounting for wraparound."""

    radians = np.deg2rad(values)
    sin_sum = np.sin(radians).mean()
    cos_sum = np.cos(radians).mean()
    angle = math.degrees(math.atan2(sin_sum, cos_sum))
    if angle < 0:
        angle += 360.0
    return angle % 360.0


def _ensure_dataframe(frame: pd.DataFrame, location: Location) -> pd.DataFrame:
    frame.insert(0, "longitude", location.longitude)
    frame.insert(0, "latitude", location.latitude)
    frame.insert(0, "location_id", location.identifier)
    return frame


def fetch_open_meteo_wind(
    location: Location,
    start_date: str | date,
    end_date: str | date,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetch daily mean wind speed and direction from the Open-Meteo API."""

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "start_date": pd.to_datetime(start_date).date().isoformat(),
        "end_date": pd.to_datetime(end_date).date().isoformat(),
        "timezone": timezone,
    }

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(
        OPEN_METEO_ENDPOINT, params=params, headers=headers, timeout=30
    )
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly", {})
    if not hourly:
        raise ValueError("Open-Meteo response did not include hourly data")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "wind_speed_kmh": hourly["wind_speed_10m"],
            "wind_dir_deg": hourly["wind_direction_10m"],
        }
    )

    frame["date"] = frame["timestamp"].dt.date
    daily = frame.groupby("date").agg(
        wind_speed_kmh=("wind_speed_kmh", "mean"),
        wind_dir_deg=("wind_dir_deg", lambda values: _circular_mean(values)),
    )

    daily.reset_index(inplace=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["wind_speed_ms"] = daily["wind_speed_kmh"] * (1000.0 / 3600.0)

    return _ensure_dataframe(daily, location)[
        [
            "location_id",
            "latitude",
            "longitude",
            "date",
            "wind_speed_kmh",
            "wind_speed_ms",
            "wind_dir_deg",
        ]
    ]


def _land_cover_payload_to_class(payload: dict) -> int:
    subset = payload.get("subset", [])
    for entry in subset:
        if entry.get("variable") == "LC_Type1":
            values = entry.get("value", [])
            if isinstance(values, Sequence) and values:
                first = values[0]
                if isinstance(first, Sequence):
                    return int(first[0])
                return int(first)
    raise ValueError("MODIS response did not include LC_Type1 values")


def _flammability_for_class(class_name: str) -> float:
    return FLAMMABILITY_SCORES.get(class_name, 0.5)


def fetch_land_cover(location: Location, year: int = 2021) -> pd.DataFrame:
    """Fetch the dominant land-cover class for the provided location."""

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "startDate": f"A{year}001",
        "endDate": f"A{year}365",
        "kmAboveBelow": 0,
        "kmLeftRight": 0,
    }

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(
        MODIS_LAND_COVER_ENDPOINT, params=params, headers=headers, timeout=30
    )
    response.raise_for_status()
    payload = response.json()

    class_id = _land_cover_payload_to_class(payload)
    class_name = IGBP_CLASSES.get(class_id, "Unknown")
    score = _flammability_for_class(class_name)

    frame = pd.DataFrame(
        {
            "land_cover_id": [class_id],
            "land_cover": [class_name],
            "flammability": [score],
        }
    )

    return _ensure_dataframe(frame, location)


def build_environment_dataset(
    locations: Iterable[Location],
    start_date: str | date,
    end_date: str | date,
    land_cover_year: int = 2021,
) -> pd.DataFrame:
    """Combine wind and land-cover information for the provided locations."""

    wind_frames: List[pd.DataFrame] = []
    cover_frames: List[pd.DataFrame] = []

    for location in locations:
        try:
            wind_frames.append(fetch_open_meteo_wind(location, start_date, end_date))
        except Exception as exc:  # pragma: no cover - depends on remote API
            LOGGER.error("Failed to fetch wind data for %s: %s", location.identifier, exc)
            raise

        try:
            cover_frames.append(fetch_land_cover(location, year=land_cover_year))
        except Exception as exc:  # pragma: no cover - depends on remote API
            LOGGER.error(
                "Failed to fetch land-cover data for %s: %s", location.identifier, exc
            )
            raise

    wind = pd.concat(wind_frames, ignore_index=True)
    cover = pd.concat(cover_frames, ignore_index=True)

    dataset = wind.merge(
        cover,
        on=["location_id", "latitude", "longitude"],
        how="left",
    )

    dataset.sort_values(["location_id", "date"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def export_environment_csv(
    output_path: str | Path,
    locations: Iterable[Location],
    start_date: str | date,
    end_date: str | date,
    land_cover_year: int = 2021,
) -> Path:
    """Fetch environmental attributes and persist them as a CSV file."""

    path = Path(output_path)
    dataset = build_environment_dataset(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        land_cover_year=land_cover_year,
    )
    dataset.to_csv(path, index=False)
    return path
