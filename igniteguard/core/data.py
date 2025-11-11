"""Data access helpers for IgniteGuard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

NASA_POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = {
    "T2M": "t2m",  # Average air temperature at 2m (°C)
    "RH2M": "rh2m",  # Relative humidity at 2m (%)
    "PRECTOTCORR": "rainfall_mm",  # Precipitation corrected (mm)
    "WS10M": "ws10m",  # Wind speed at 10m (m/s)
    "ALLSKY_SFC_SW_DWN": "srad",  # Shortwave radiation (MJ/m^2/day)
}

DEFAULT_OFFLINE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_input.csv"


def _normalise_date(value: str) -> str:
    """Normalise user-provided dates to the YYYYMMDD format required by NASA."""

    return pd.to_datetime(value).strftime("%Y%m%d")


def _build_frame(parameters: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Transform the NASA POWER parameter response into a tidy DataFrame."""

    records: Dict[str, Dict[str, float]] = {}
    for source_key, target_key in NASA_PARAMETERS.items():
        values = parameters.get(source_key, {})
        for date_str, value in values.items():
            records.setdefault(date_str, {})[target_key] = value

    if not records:
        raise ValueError("NASA POWER response did not include expected parameters")

    frame = pd.DataFrame.from_dict(records, orient="index")
    frame.index.name = "date"
    frame.sort_index(inplace=True)
    frame.reset_index(inplace=True)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")

    # Ensure optional fields exist for downstream consumers.
    for optional_col in ("ndvi", "ndwi", "soil_moisture"):
        if optional_col not in frame.columns:
            frame[optional_col] = pd.NA

    return frame


def get_power_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch daily aggregates from the NASA POWER API.

    The function automatically falls back to the offline CSV dataset when the
    remote request fails for any reason (network error, unexpected payload,
    status code, etc.).
    """

    params = {
        "start": _normalise_date(start),
        "end": _normalise_date(end),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "format": "JSON",
        "parameters": ",".join(NASA_PARAMETERS.keys()),
    }

    try:
        response = requests.get(NASA_POWER_ENDPOINT, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()
        parameters = payload["properties"]["parameter"]
        return _build_frame(parameters)
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning(
            "Falling back to offline dataset after NASA POWER error: %s", exc
        )
        return load_offline_csv(DEFAULT_OFFLINE_CSV)


def load_offline_csv(path: str | Path) -> pd.DataFrame:
    """Load offline observations from a CSV file.

    The CSV is expected to contain the following columns:

    ``date, t2m, rh2m, rainfall_mm, ws10m, srad, ndvi, ndwi, soil_moisture``
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Offline CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, parse_dates=["date"])
    required_columns: Iterable[str] = (
        "date",
        "t2m",
        "rh2m",
        "rainfall_mm",
        "ws10m",
        "srad",
        "ndvi",
        "ndwi",
        "soil_moisture",
    )

    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise ValueError(
            f"Offline CSV is missing required columns: {', '.join(missing)}"
        )

    frame = frame[list(required_columns)].copy()
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame
