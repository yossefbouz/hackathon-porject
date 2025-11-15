"""Tests for the environment data acquisition helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from igniteguard.core.environment import (
    Location,
    build_environment_dataset,
    export_environment_csv,
    fetch_land_cover,
    fetch_open_meteo_wind,
)


class DummyResponse:
    """Simple stand-in for ``requests.Response``."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture()
def downtown_la() -> Location:
    return Location(identifier="la", latitude=34.05, longitude=-118.24)


def _mock_wind_payload() -> dict:
    return {
        "hourly": {
            "time": [
                "2024-05-01T00:00",
                "2024-05-01T01:00",
                "2024-05-02T00:00",
                "2024-05-02T01:00",
            ],
            "wind_speed_10m": [10.0, 14.0, 20.0, 22.0],
            "wind_direction_10m": [350.0, 10.0, 90.0, 120.0],
        }
    }


def _mock_land_cover_payload() -> dict:
    return {
        "subset": [
            {
                "variable": "LC_Type1",
                "value": [10],
            }
        ]
    }


def test_fetch_open_meteo_wind(monkeypatch: pytest.MonkeyPatch, downtown_la: Location) -> None:
    def fake_get(*_: object, **__: object) -> DummyResponse:
        return DummyResponse(_mock_wind_payload())

    monkeypatch.setattr("requests.get", fake_get)

    frame = fetch_open_meteo_wind(downtown_la, "2024-05-01", "2024-05-02")

    assert list(frame.columns) == [
        "location_id",
        "latitude",
        "longitude",
        "date",
        "wind_speed_kmh",
        "wind_speed_ms",
        "wind_dir_deg",
    ]
    assert frame.shape == (2, 7)
    assert pytest.approx(frame.loc[0, "wind_speed_kmh"], abs=0.01) == 12.0
    # 10 km/h to m/s -> 2.777...
    assert pytest.approx(frame.loc[0, "wind_speed_ms"], abs=0.01) == 3.33
    # Circular mean of 350 and 10 should be 0 (north)
    assert pytest.approx(frame.loc[0, "wind_dir_deg"], abs=0.01) == 0.0


def test_fetch_land_cover(monkeypatch: pytest.MonkeyPatch, downtown_la: Location) -> None:
    def fake_get(*_: object, **__: object) -> DummyResponse:
        return DummyResponse(_mock_land_cover_payload())

    monkeypatch.setattr("requests.get", fake_get)

    frame = fetch_land_cover(downtown_la, year=2021)

    assert frame.loc[0, "land_cover_id"] == 10
    assert frame.loc[0, "land_cover"] == "Grasslands"
    assert frame.loc[0, "flammability"] == 0.9


def test_build_environment_dataset(monkeypatch: pytest.MonkeyPatch, downtown_la: Location) -> None:
    responses: Iterator[DummyResponse] = iter(
        [
            DummyResponse(_mock_wind_payload()),
            DummyResponse(_mock_land_cover_payload()),
        ]
    )

    def fake_get(*_: object, **__: object) -> DummyResponse:
        return next(responses)

    monkeypatch.setattr("requests.get", fake_get)

    dataset = build_environment_dataset(
        locations=[downtown_la],
        start_date="2024-05-01",
        end_date="2024-05-02",
        land_cover_year=2021,
    )

    assert dataset.shape == (2, 10)
    assert set(dataset.columns) == {
        "location_id",
        "latitude",
        "longitude",
        "date",
        "wind_speed_kmh",
        "wind_speed_ms",
        "wind_dir_deg",
        "land_cover_id",
        "land_cover",
        "flammability",
    }


def test_export_environment_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, downtown_la: Location) -> None:
    responses: Iterator[DummyResponse] = iter(
        [
            DummyResponse(_mock_wind_payload()),
            DummyResponse(_mock_land_cover_payload()),
        ]
    )

    def fake_get(*_: object, **__: object) -> DummyResponse:
        return next(responses)

    monkeypatch.setattr("requests.get", fake_get)

    output_file = tmp_path / "environment.csv"
    export_environment_csv(
        output_file,
        locations=[downtown_la],
        start_date=date(2024, 5, 1),
        end_date=date(2024, 5, 2),
        land_cover_year=2021,
    )

    assert output_file.exists()
    frame = pd.read_csv(output_file, parse_dates=["date"])
    assert list(frame["land_cover"].unique()) == ["Grasslands"]
