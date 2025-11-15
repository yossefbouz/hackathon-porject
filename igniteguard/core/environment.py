"""External data acquisition helpers for wind statistics."""

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

USER_AGENT = "IgniteGuard/0.1 (+https://example.com/igniteguard)"


@dataclass(frozen=True)
class Location:
    """Simple geographic coordinate container."""

    identifier: str
    latitude: float
    longitude: float


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

def build_environment_dataset(
    locations: Iterable[Location],
    start_date: str | date,
    end_date: str | date,
) -> pd.DataFrame:
    """Fetch Open-Meteo wind statistics for the provided locations."""

    wind_frames: List[pd.DataFrame] = []

    for location in locations:
        try:
            wind_frames.append(fetch_open_meteo_wind(location, start_date, end_date))
        except Exception as exc:  # pragma: no cover - depends on remote API
            LOGGER.error("Failed to fetch wind data for %s: %s", location.identifier, exc)
            raise

    wind = pd.concat(wind_frames, ignore_index=True)

    wind.sort_values(["location_id", "date"], inplace=True)
    wind.reset_index(drop=True, inplace=True)
    return wind


def export_environment_csv(
    output_path: str | Path,
    locations: Iterable[Location],
    start_date: str | date,
    end_date: str | date,
) -> Path:
    """Fetch environmental attributes and persist them as a CSV file."""

    path = Path(output_path)
    dataset = build_environment_dataset(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
    )
    dataset.to_csv(path, index=False)
    return path
