"""IgniteGuard Streamlit dashboard for Limassol dryness monitoring."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA_PATH = "igniteguard_last30days.csv"
DEFAULT_LAT = 34.707
DEFAULT_LON = 33.03

def set_page() -> None:
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="IgniteGuard – Limassol Dryness Dashboard",
        layout="wide",
    )

@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and standardise the Limassol dryness dataset."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(
            "CSV file 'igniteguard_last30days.csv' was not found. "
            "Ensure it is located next to app.py."
        )
        st.stop()
    df.columns = df.columns.str.strip()
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df["display_date"] = df[date_col].dt.date
    else:
        df = df.reset_index(drop=True)
        df["display_date"] = df.index
    for col in df.columns:
        if col == "display_date" or col in date_cols:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def compute_dryness(df: pd.DataFrame) -> pd.DataFrame:
    """Compute dryness index and risk band based on flammability/NDVI/NDWI."""
    df = df.copy()
    flammability = df.get("flammability")
    ndvi = df.get("ndvi", pd.Series(np.nan, index=df.index))
    ndwi = df.get("ndwi", pd.Series(np.nan, index=df.index))

    if flammability is not None and not flammability.isna().all():
        dryness = flammability.astype(float) * 100
        fallback_mask = dryness.isna()
        fallback = ((-ndwi + (1 - ndvi)) / 2) * 100
        dryness[fallback_mask] = fallback[fallback_mask]
    else:
        dryness = ((-ndwi + (1 - ndvi)) / 2) * 100
    dryness = dryness.clip(lower=0, upper=100)
    df["dryness_index"] = dryness
    return df

def assign_risk_band(df: pd.DataFrame, safe_thr: float, danger_thr: float) -> pd.DataFrame:
    """Assign qualitative risk band labels given thresholds."""
    df = df.copy()
    conditions = [
        df["dryness_index"] < safe_thr,
        (df["dryness_index"] >= safe_thr) & (df["dryness_index"] < danger_thr),
        df["dryness_index"] >= danger_thr,
    ]
    choices = ["Safe", "Watch", "Danger"]
    df["risk_band"] = np.select(conditions, choices, default="Safe")
    return df

@dataclass
class Filters:
    selected_dates: Iterable
    selected_bands: Iterable[str]
    safe_threshold: float
    danger_threshold: float

def sidebar_controls(df: pd.DataFrame) -> Filters:
    """Render sidebar controls and return the applied filters."""
    st.sidebar.header("Controls")
    safe_default = 40
    danger_default = 70
    safe_thr = st.sidebar.slider(
        "Safe / Watch boundary",
        min_value=0,
        max_value=90,
        value=safe_default,
        step=1,
    )
    danger_thr = st.sidebar.slider(
        "Watch / Danger boundary",
        min_value=safe_thr + 1,
        max_value=100,
        value=max(danger_default, safe_thr + 1),
        step=1,
    )
    unique_dates = df["display_date"].unique().tolist()
    latest_date = unique_dates[-1] if unique_dates else None
    if latest_date is not None and isinstance(latest_date, (pd.Timestamp, datetime, date)):
        date_label = "Date"
    else:
        date_label = "Index"
    default_dates = [latest_date] if latest_date is not None else unique_dates
    selected_dates = st.sidebar.multiselect(
        f"Select {date_label.lower()}(s)",
        options=unique_dates,
        default=default_dates,
    )
    available_bands = [
        band
        for band in df.get("risk_band", pd.Series(dtype=str)).dropna().unique().tolist()
        if band
    ]
    if not available_bands:
        available_bands = ["Safe", "Watch", "Danger"]
    selected_bands = st.sidebar.multiselect(
        "Risk bands",
        options=available_bands,
        default=available_bands,
    )
    return Filters(
        selected_dates=selected_dates,
        selected_bands=selected_bands,
        safe_threshold=safe_thr,
        danger_threshold=danger_thr,
    )

def latest_record(df: pd.DataFrame) -> pd.Series:
    """Return the latest record based on display ordering."""
    if "display_date" in df.columns:
        sorted_df = df.sort_values(by="display_date")
    else:
        sorted_df = df.reset_index(drop=True)
    return sorted_df.iloc[-1]

def display_metrics(df: pd.DataFrame) -> None:
    """Show summary metric cards."""
    latest = latest_record(df)
    col1, col2, col3 = st.columns(3)
    dryness_value = latest.get("dryness_index")
    col1.metric(
        "Latest Dryness Index",
        "N/A" if pd.isna(dryness_value) else f"{dryness_value:.0f}",
    )
    col2.metric("Risk Band", latest.get("risk_band", "N/A"))
    if "wind_speed" in df:
        wind_avg = df["wind_speed"].mean()
        col3.metric(
            "Average Wind Speed (last 30 days)",
            "N/A" if pd.isna(wind_avg) else f"{wind_avg:.1f} m/s",
        )
    else:
        col3.metric("Average Wind Speed (last 30 days)", "N/A")

def make_time_series(df: pd.DataFrame, date_column: str) -> None:
    """Render the dryness time-series chart."""
    if date_column not in df.columns:
        st.info("No date column found to build time series.")
        return
    chart_df = df[[date_column, "dryness_index"]].copy()
    chart_df = chart_df.rename(columns={date_column: "Date"})
    chart_df["Dryness Index"] = chart_df["dryness_index"]
    lines = ["Dryness Index"]
    if "ndvi" in df.columns:
        chart_df["NDVI (scaled)"] = ((df["ndvi"] + 1) / 2 * 100).clip(0, 100)
        lines.append("NDVI (scaled)")
    if "ndwi" in df.columns:
        chart_df["NDWI (scaled)"] = ((df["ndwi"] + 1) / 2 * 100).clip(0, 100)
        lines.append("NDWI (scaled)")
    chart_df = chart_df[["Date", *lines]]
    chart_df = chart_df.melt("Date", var_name="Series", value_name="Value")
    date_series = chart_df["Date"]
    x_field = alt.X("Date:T", title="Date")
    tooltip_fields = [alt.Tooltip("Series:N", title="Series"), alt.Tooltip("Value:Q", title="Value", format=".1f")]
    if not np.issubdtype(date_series.dtype, np.datetime64):
        try:
            converted = pd.to_datetime(date_series)
            if converted.notna().any():
                chart_df["Date"] = converted
            else:
                raise ValueError
        except (ValueError, TypeError):
            chart_df["Date Label"] = date_series.astype(str)
            x_field = alt.X("Date Label:N", title="Date / Index")
            tooltip_fields.insert(0, alt.Tooltip("Date Label:N", title="Date / Index"))
        else:
            tooltip_fields.insert(0, alt.Tooltip("Date:T", title="Date"))
    else:
        tooltip_fields.insert(0, alt.Tooltip("Date:T", title="Date"))
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=x_field,
            y=alt.Y("Value:Q", title="Index"),
            color="Series:N",
            tooltip=tooltip_fields,
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

def risk_color(band: str) -> list[int]:
    mapping = {
        "Safe": [34, 139, 34],
        "Watch": [255, 215, 0],
        "Danger": [220, 20, 60],
    }
    return mapping.get(band, [150, 150, 150])

def make_heatmap(df: pd.DataFrame) -> Optional[pdk.Deck]:
    """Create a pydeck map visualising dryness index and risk band."""
    if df.empty:
        return None
    risk_series = df.get("risk_band", pd.Series("Safe", index=df.index))
    scatter_data = df.assign(color=risk_series.apply(risk_color), risk_band=risk_series)
    if "lon" not in df.columns or "lat" not in df.columns:
        return None
    layers = [
        pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position="[lon, lat]",
            get_weight="dryness_index",
            radiusPixels=60,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=scatter_data,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=120,
            pickable=True,
            radius_scale=1,
        ),
    ]
    lon_series = df.get("lon")
    lat_series = df.get("lat")
    longitude = float(lon_series.dropna().iloc[0]) if lon_series is not None and not lon_series.dropna().empty else DEFAULT_LON
    latitude = float(lat_series.dropna().iloc[0]) if lat_series is not None and not lat_series.dropna().empty else DEFAULT_LAT
    view_state = pdk.ViewState(
        longitude=longitude,
        latitude=latitude,
        zoom=10,
        pitch=40,
    )
    tooltip = {
        "html": "<b>{display_date}</b><br/>Dryness: {dryness_index:.1f}<br/>Risk: {risk_band}",
        "style": {"backgroundColor": "#1f2630", "color": "white"},
    }
    return pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)

def degrees_to_cardinal(deg: float) -> str:
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    idx = int((deg % 360) / 22.5 + 0.5) % 16
    return directions[idx]

def display_conditions(df: pd.DataFrame) -> None:
    """Show wind and slope details for the selected period."""
    if df.empty:
        st.info("No data available for selected filters.")
        return
    latest = latest_record(df)
    st.subheader("Conditions")
    cols = st.columns(3)
    if "wind_speed" in latest:
        cols[0].metric("Wind Speed", f"{latest['wind_speed']:.1f} m/s")
    if "wind_dir" in latest:
        direction = degrees_to_cardinal(latest["wind_dir"])
        cols[1].metric("Wind Direction", f"{latest['wind_dir']:.0f}° ({direction})")
    if "slope" in latest:
        cols[2].metric("Slope", f"{latest['slope']:.1f}°")

def land_cover_breakdown(df: pd.DataFrame) -> None:
    """Render a land cover dryness/flammability summary."""
    if "land_cover" not in df.columns:
        st.info("Land cover data unavailable.")
        return
    agg_spec: dict[str, tuple[str, str]] = {"count": ("land_cover", "size")}
    if "flammability" in df.columns:
        agg_spec["avg_flammability"] = ("flammability", "mean")
    if "dryness_index" in df.columns:
        agg_spec["avg_dryness"] = ("dryness_index", "mean")
    group = df.groupby("land_cover").agg(**agg_spec).reset_index()
    if group.empty:
        st.info("No land cover observations to summarise.")
        return
    if "avg_flammability" not in group.columns:
        group["avg_flammability"] = np.nan
    if "avg_dryness" not in group.columns:
        group["avg_dryness"] = np.nan
    group[["avg_flammability", "avg_dryness"]] = group[["avg_flammability", "avg_dryness"]].fillna(0)
    chart = (
        alt.Chart(group)
        .mark_bar()
        .encode(
            x=alt.X("land_cover:N", title="Land Cover"),
            y=alt.Y("avg_dryness:Q", title="Avg Dryness Index"),
            color=alt.Color("avg_flammability:Q", title="Avg Flammability", scale=alt.Scale(scheme="reds")),
            tooltip=[
                "land_cover",
                alt.Tooltip("count:Q", title="Observations"),
                alt.Tooltip("avg_dryness:Q", title="Avg Dryness", format=".1f"),
                alt.Tooltip("avg_flammability:Q", title="Avg Flammability", format=".2f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

def top_dangerous_days(df: pd.DataFrame) -> None:
    """Display a table of the top 5 driest days."""
    if df.empty:
        return
    st.subheader("Top 5 Most Dangerous Days")
    display_cols = [c for c in ["display_date", "dryness_index", "risk_band", "wind_speed", "slope"] if c in df.columns]
    table = df.sort_values("dryness_index", ascending=False).head(5)[display_cols]
    table = table.rename(columns={"display_date": "Date"})
    st.table(table)

def explanation_panel() -> None:
    """Explain dryness and risk calculations."""
    with st.expander("How we compute dryness & risk"):
        st.markdown(
            """
            **Dryness index** comes from the pre-computed flammability score when available\n
            (`dryness_index = flammability × 100`, clipped to the 0–100 range). When
            flammability is absent, the app estimates dryness from the vegetation and water
            balance (`ndvi`, `ndwi`) captured by Sentinel-2 imagery over the Limassol (Polimedia)
            monitoring footprint.

            **Risk bands** translate the dryness index into qualitative alerts:

            - **Safe:** dryness index below the Safe/Watch threshold (default 40).
            - **Watch:** dryness between the Safe/Watch and Watch/Danger thresholds (default 70).
            - **Danger:** dryness index above the Watch/Danger threshold.

            Sentinel-2 derived NDVI/NDWI cover the red-highlighted Limassol district where the
            latest large wildfire occurred, allowing the dashboard to mirror real vegetation
            stress dynamics over the past 30 days.
            """
        )

def apply_filters(df: pd.DataFrame, filters: Filters, date_column: str) -> pd.DataFrame:
    """Apply sidebar filters to the dataframe."""
    filtered = df
    if filters.selected_dates:
        filtered = filtered[filtered["display_date"].isin(filters.selected_dates)]
    if filters.selected_bands:
        filtered = filtered[filtered["risk_band"].isin(filters.selected_bands)]
    if date_column and date_column in filtered.columns:
        filtered = filtered.sort_values(date_column)
    return filtered

def get_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if "date" in c.lower() and c != "display_date"]
    return candidates[0] if candidates else None

def main() -> None:
    set_page()
    st.title("IgniteGuard – Limassol Dryness & Fire Spread Prototype")
    st.caption("Real NDVI/NDWI-based dryness model for Limassol (Polimedia area), last 30 days.")

    df = load_data()
    df = compute_dryness(df)
    date_col = get_date_column(df)

    sidebar_df = assign_risk_band(df, 40, 70)
    filters = sidebar_controls(sidebar_df)
    df = assign_risk_band(df, filters.safe_threshold, filters.danger_threshold)

    with st.expander("Raw data (debug)"):
        st.dataframe(df.head())

    display_metrics(df)

    filtered_df = apply_filters(df, filters, date_col if date_col else "")

    st.subheader("Dryness Trend")
    time_series_df = df if date_col else df.reset_index().rename(columns={"index": "Index"})
    make_time_series(time_series_df, date_col or "Index")

    st.subheader("Dryness Heat Map")
    deck = make_heatmap(filtered_df)
    if deck is None:
        st.info("No data available for the selected filters to display on the map.")
    else:
        st.pydeck_chart(deck)

    display_conditions(filtered_df)

    st.subheader("Land Cover & Flammability")
    land_cover_breakdown(df)

    top_dangerous_days(df)

    explanation_panel()

if __name__ == "__main__":
    main()
