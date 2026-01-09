"""Symbol universes (S&P 500, Nasdaq-100) and helpers.

Important:
- Index constituents change over time; this is a convenience helper.
- Wikipedia scraping requires either lxml or html5lib for pandas.read_html.
  If it's not available, use --symbols or --symbols-file instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


def load_symbols_file(path: str | Path) -> list[str]:
    """Load symbols from a .txt or .csv file.

    - .txt: one symbol per line
    - .csv: column named 'symbol' or the first column
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() in {".txt", ".list"}:
        syms: list[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip().upper()
            if s and not s.startswith("#"):
                syms.append(s)
        return syms

    df = pd.read_csv(p)
    if df.empty:
        return []

    col = None
    for c in df.columns:
        if str(c).strip().lower() in {"symbol", "ticker", "tickers"}:
            col = c
            break
    if col is None:
        col = df.columns[0]

    return [str(x).strip().upper() for x in df[col].dropna().tolist() if str(x).strip()]


def _fetch_html(url: str) -> str:
    r = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.text


def fetch_sp500_symbols() -> list[str]:
    """Best-effort fetch of current S&P 500 constituents."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = _fetch_html(url)
    try:
        tables = pd.read_html(html)
    except Exception as e:
        raise RuntimeError(
            "Unable to parse S&P 500 table. Install 'lxml' (recommended) or provide --symbols-file instead."
        ) from e

    # The first table usually contains the constituents with 'Symbol'
    for t in tables:
        cols = {str(c).strip().lower(): c for c in t.columns}
        if "symbol" in cols:
            s = t[cols["symbol"]].astype(str).str.strip().str.upper().tolist()
            return [x.replace(".", "-") for x in s if x]

    raise RuntimeError("Could not find 'Symbol' column on S&P 500 page")


def fetch_nasdaq100_symbols() -> list[str]:
    """Best-effort fetch of current Nasdaq-100 constituents."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = _fetch_html(url)
    try:
        tables = pd.read_html(html)
    except Exception as e:
        raise RuntimeError(
            "Unable to parse Nasdaq-100 table. Install 'lxml' (recommended) or provide --symbols-file instead."
        ) from e

    # One of the tables has 'Ticker'
    for t in tables:
        cols = {str(c).strip().lower(): c for c in t.columns}
        for key in ("ticker", "symbol"):
            if key in cols:
                s = t[cols[key]].astype(str).str.strip().str.upper().tolist()
                return [x.replace(".", "-") for x in s if x]

    raise RuntimeError("Could not find 'Ticker'/'Symbol' column on Nasdaq-100 page")
