"""Macro / fundamental data sources (FRED, World Bank).

These are lightweight HTTP clients with minimal dependencies.
- FRED: https://fred.stlouisfed.org/docs/api/fred/
- World Bank Open Data: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

Notes:
- Some endpoints may rate-limit; cache outputs on disk when practical.
- This module intentionally returns pandas Series/DataFrames indexed by date,
  so callers can merge/ffill onto market trading calendars.

Research/education only; not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import os

import pandas as pd
import requests


@dataclass(frozen=True)
class FREDSeriesRequest:
    series_id: str
    start: str | None = None  # YYYY-MM-DD
    end: str | None = None    # YYYY-MM-DD
    api_key: str | None = None


def fetch_fred_series(req: FREDSeriesRequest) -> pd.Series:
    """Fetch a single FRED series as a pandas Series.

    Returns:
      Series indexed by Timestamp (date), dtype float.

    Raises:
      requests.HTTPError on non-2xx responses.
      ValueError if payload is missing expected fields.
    """
    series_id = (req.series_id or "").strip()
    if not series_id:
        raise ValueError("FRED series_id is required")

    api_key = (req.api_key or os.getenv("FRED_API_KEY") or "").strip() or None

    url = "https://api.stlouisfed.org/fred/series/observations"
    params: dict[str, str] = {
        "series_id": series_id,
        "file_type": "json",
    }
    if api_key:
        params["api_key"] = api_key
    if req.start:
        params["observation_start"] = str(req.start)
    if req.end:
        params["observation_end"] = str(req.end)

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations")
    if not isinstance(obs, list):
        raise ValueError(f"Unexpected FRED response for {series_id}: missing observations")

    rows = []
    for o in obs:
        dt = o.get("date")
        val = o.get("value")
        rows.append((dt, val))

    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    s = df["value"].astype(float)
    s.name = series_id
    return s


def fetch_fred_many(
    series_ids: Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch multiple FRED series into a single DataFrame (outer-joined by date)."""
    frames: list[pd.Series] = []
    for sid in series_ids:
        sid = (sid or "").strip()
        if not sid:
            continue
        frames.append(fetch_fred_series(FREDSeriesRequest(series_id=sid, start=start, end=end, api_key=api_key)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


@dataclass(frozen=True)
class WorldBankRequest:
    indicator: str              # e.g. FP.CPI.TOTL.ZG
    country: str = "US"         # ISO2 or 'all'
    start: str | None = None    # YYYY
    end: str | None = None      # YYYY


def fetch_world_bank_indicator(req: WorldBankRequest) -> pd.Series:
    """Fetch a World Bank indicator time series.

    Returns annual data as a Series indexed by Timestamp (year start).

    World Bank returns years as strings; we map them to Timestamps of Jan 1.
    """
    indicator = (req.indicator or "").strip()
    if not indicator:
        raise ValueError("World Bank indicator is required")

    country = (req.country or "US").strip() or "US"
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

    params: dict[str, str] = {
        "format": "json",
        "per_page": "20000",
    }
    if req.start:
        params["date"] = str(req.start) if not req.end else f"{req.start}:{req.end}"

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Unexpected World Bank response")

    data = payload[1]
    if not isinstance(data, list):
        raise ValueError("Unexpected World Bank response (data)")

    rows = []
    for item in data:
        year = item.get("date")
        value = item.get("value")
        rows.append((year, value))

    df = pd.DataFrame(rows, columns=["year", "value"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year"]).sort_values("year")
    df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-01-01", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")

    s = df["value"].astype(float)
    s.name = f"wb_{country}_{indicator}"
    return s


def fetch_world_bank_many(requests_list: Iterable[WorldBankRequest]) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for r in requests_list:
        frames.append(fetch_world_bank_indicator(r))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()
