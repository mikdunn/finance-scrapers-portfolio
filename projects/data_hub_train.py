"""Data hub: build a combined dataset from multiple sources and train.

Supported sources (best-effort):
- Market prices via existing provider chain (Yahoo -> Kraken/Coinbase -> Stooq)
- FRED series (macro) via HTTP API
- World Bank indicators (annual macro)
- Local CSVs (Kaggle/AEA/etc.) as exogenous features (by date)

This writes per-symbol CSVs into an output directory, then runs the existing ML
trainer in multi-asset mode.

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np

from utils.market_data import OHLCVRequest, fetch_ohlcv
from utils.macro_data import FREDSeriesRequest, WorldBankRequest, fetch_fred_many, fetch_world_bank_many
from utils.universe import fetch_nasdaq100_symbols, fetch_sp500_symbols, load_symbols_file
from utils.spectral import spectral_cluster_assets
from utils.dmd import exact_dmd, eigenvalue_to_frequency
from utils.portfolio import cluster_inverse_vol_allocation


def _arima_forecast(close: pd.Series, steps: int = 5) -> list[float]:
    """ARIMA forecast of closing price.

    Uses statsmodels if available. Falls back to a naive last-value forecast.
    Kept local to the hub so chart generation doesn't depend on importing
    projects.market_analyzer.
    """
    close = close.dropna().astype(float)
    if steps <= 0:
        return []
    if len(close) < 50:
        return [float(close.iloc[-1])] * steps if len(close) else []

    try:
        import warnings
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = ARIMA(close, order=(1, 1, 1))
            fitted = model.fit()
            fc = fitted.forecast(steps=steps)
            return [float(x) for x in fc.values]
    except Exception:
        return [float(close.iloc[-1])] * steps


def _is_equity_symbol(symbol: str) -> bool:
    s = (symbol or '').strip().upper()
    if not s:
        return False
    # crude heuristic: equities are simple tickers like AAPL, BRK.B, etc.
    if any(x in s for x in ['-', '=']):
        return False
    return True


def _parse_rss_dates(xml_bytes: bytes) -> list[datetime]:
    """Parse RSS/Atom and return item timestamps as naive UTC datetimes when possible."""
    root = ET.fromstring(xml_bytes)
    dates: list[datetime] = []

    # RSS
    for item in root.findall('.//item'):
        dt_txt = None
        pub = item.find('pubDate')
        if pub is not None and pub.text:
            dt_txt = pub.text.strip()
        if not dt_txt:
            continue
        try:
            # Example: Tue, 09 Jan 2026 15:42:00 GMT
            from email.utils import parsedate_to_datetime

            d = parsedate_to_datetime(dt_txt)
            if d is None:
                continue
            if getattr(d, 'tzinfo', None) is not None:
                d = d.astimezone(tz=None).replace(tzinfo=None)
            dates.append(d)
        except Exception:
            continue

    # Atom
    if not dates:
        for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
            upd = entry.find('{http://www.w3.org/2005/Atom}updated')
            if upd is None or not upd.text:
                continue
            txt = upd.text.strip()
            try:
                # ISO8601; best-effort
                # Handle Z suffix
                txt2 = txt.replace('Z', '+00:00')
                d = datetime.fromisoformat(txt2)
                if getattr(d, 'tzinfo', None) is not None:
                    d = d.astimezone(tz=None).replace(tzinfo=None)
                dates.append(d)
            except Exception:
                continue

    return dates


def _news_hit(symbol: str, *, days: int = 3, min_items: int = 1, timeout: int = 15) -> tuple[bool, int]:
    """Return (is_in_news, recent_item_count) using Google News RSS.

    For speed/reliability this uses RSS only (no Selenium). This is a heuristic.
    """
    import requests

    ticker = (symbol or '').strip().upper()
    if not ticker:
        return False, 0

    # Google News RSS search
    url = (
        'https://news.google.com/rss/search?'
        f'q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en'
    )

    try:
        r = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        r.raise_for_status()
        dates = _parse_rss_dates(r.content)
        if dates:
            cutoff = datetime.now() - timedelta(days=int(days))
            recent = [d for d in dates if d >= cutoff]
            return (len(recent) >= int(min_items)), int(len(recent))

        # If dates not parseable, fall back to counting titles.
        xml_text = r.content.decode('utf-8', errors='ignore')
        # crude: count <item> tags
        n_items = len(re.findall(r'<item\b', xml_text, flags=re.IGNORECASE))
        return (n_items >= int(min_items)), int(n_items)
    except Exception:
        return False, 0


def _fib_retracement_levels(swing_low: float, swing_high: float) -> dict[str, float]:
    """Return common Fibonacci retracement levels for an up-move (low->high).

    Levels are prices where an uptrend pullback might find support.
    """
    r = float(swing_high) - float(swing_low)
    if r <= 0:
        return {}
    return {
        '0.236': float(swing_high) - 0.236 * r,
        '0.382': float(swing_high) - 0.382 * r,
        '0.500': float(swing_high) - 0.500 * r,
        '0.618': float(swing_high) - 0.618 * r,
        '0.786': float(swing_high) - 0.786 * r,
    }


def _fib_check(
    df: pd.DataFrame,
    *,
    direction: str,
    lookback: int = 60,
    tolerance: float = 0.01,
) -> tuple[bool, dict[str, float] | None, str | None]:
    """Check whether price is near a meaningful fib retracement level.

    Returns (pass, levels, nearest_level_key).
    """
    if df is None or df.empty or len(df) < 5:
        return False, None, None

    n = int(lookback)
    if n <= 5:
        n = min(60, len(df))
    window = df.iloc[-n:] if len(df) >= n else df

    hi = float(pd.to_numeric(window['High'], errors='coerce').max())
    lo = float(pd.to_numeric(window['Low'], errors='coerce').min())
    if not (pd.notna(hi) and pd.notna(lo)):
        return False, None, None
    if hi <= lo:
        return False, None, None

    close = float(pd.to_numeric(df['Close'].iloc[-1], errors='coerce'))
    if not pd.notna(close) or close <= 0:
        return False, None, None

    direction = (direction or '').strip().lower()

    # Compute levels for uptrend pullback (low->high).
    levels = _fib_retracement_levels(lo, hi)
    if not levels:
        return False, None, None

    # For shorts, we still use the same numeric levels but interpret them as potential
    # resistance on a bounce in a downtrend.
    tol = float(tolerance)
    nearest_k = None
    nearest_dist = None
    for k, lvl in levels.items():
        dist = abs(close - float(lvl)) / close
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_k = k

    passes = bool(nearest_dist is not None and nearest_dist <= tol)
    return passes, levels, nearest_k


def _ichimoku_trade_signal(df: pd.DataFrame) -> tuple[bool, str | None, dict[str, float] | None]:
    """Heuristic Ichimoku pass/fail + direction.

    Returns (pass, direction, debug_values).
    """
    from utils.technical import ichimoku

    if df is None or df.empty or len(df) < 80:
        return False, None, None

    ichi = ichimoku(df['High'], df['Low'], df['Close'])
    close = float(df['Close'].iloc[-1])
    tenkan = float(ichi.tenkan_sen.iloc[-1]) if pd.notna(ichi.tenkan_sen.iloc[-1]) else None
    kijun = float(ichi.kijun_sen.iloc[-1]) if pd.notna(ichi.kijun_sen.iloc[-1]) else None
    span_a = float(ichi.senkou_span_a.iloc[-1]) if pd.notna(ichi.senkou_span_a.iloc[-1]) else None
    span_b = float(ichi.senkou_span_b.iloc[-1]) if pd.notna(ichi.senkou_span_b.iloc[-1]) else None

    if tenkan is None or kijun is None or span_a is None or span_b is None:
        return False, None, None

    cloud_top = max(span_a, span_b)
    cloud_bot = min(span_a, span_b)

    long_ok = (close > cloud_top) and (tenkan > kijun) and (span_a >= span_b)
    short_ok = (close < cloud_bot) and (tenkan < kijun) and (span_a <= span_b)

    direction = 'long' if long_ok else ('short' if short_ok else None)
    passed = bool(long_ok or short_ok)
    return passed, direction, {
        'close': close,
        'tenkan': tenkan,
        'kijun': kijun,
        'span_a': span_a,
        'span_b': span_b,
        'cloud_top': cloud_top,
        'cloud_bottom': cloud_bot,
    }


def _rsi_trade_ok(df: pd.DataFrame, *, direction: str | None) -> tuple[bool, float | None]:
    from utils.technical import rsi

    if df is None or df.empty or len(df) < 20:
        return False, None
    r = rsi(df['Close'], 14)
    if r.dropna().empty:
        return False, None
    val = float(r.dropna().iloc[-1])
    direction = (direction or '').strip().lower()
    if direction == 'long':
        return (50.0 <= val <= 70.0), val
    if direction == 'short':
        return (30.0 <= val <= 50.0), val
    # if no direction, accept neutral-ish RSI as a weak filter
    return (45.0 <= val <= 55.0), val


def _select_and_copy_charts(
    *,
    out_dir: Path,
    charts_dir: Path,
    scored: list[dict],
    min_conditions: int,
    max_selected: int | None,
) -> dict:
    """Write a selection manifest and copy chosen chart files into selected_charts/."""
    min_conditions = int(min_conditions)
    max_selected_i = int(max_selected) if max_selected is not None else None

    eligible = [x for x in scored if int(x.get('conditions_met', 0)) >= min_conditions]
    eligible.sort(key=lambda d: (int(d.get('conditions_met', 0)), float(d.get('score', 0.0))), reverse=True)
    if max_selected_i is not None and max_selected_i > 0:
        eligible = eligible[:max_selected_i]

    selected_dir = out_dir / 'selected_charts'
    selected_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    for item in eligible:
        arts = (item.get('artifacts') or {})
        html_src = arts.get('html')
        csv_src = arts.get('csv')
        if html_src:
            try:
                dst = selected_dir / Path(str(html_src)).name
                shutil.copy2(html_src, dst)
                item.setdefault('selected_files', {})['html'] = str(dst)
            except Exception:
                pass
        if csv_src:
            try:
                dst = selected_dir / Path(str(csv_src)).name
                shutil.copy2(csv_src, dst)
                item.setdefault('selected_files', {})['csv'] = str(dst)
            except Exception:
                pass
        copied.append(item)

    selection_manifest = {
        'min_conditions': min_conditions,
        'max_selected': max_selected_i,
        'charts_dir': str(charts_dir),
        'selected_dir': str(selected_dir),
        'selected': copied,
        'not_selected_count': max(0, len(scored) - len(copied)),
    }
    with open(out_dir / 'selected_charts.json', 'w', encoding='utf-8') as f:
        json.dump(selection_manifest, f, indent=2)
    return selection_manifest


def _align_closes(closes: dict[str, pd.Series], *, how: str = 'inner') -> pd.DataFrame:
    """Align close series into a single DataFrame indexed by datetime.

    closes: {symbol: close_series}
    how: 'inner' (intersection) or 'outer' (union, then forward-fill)
    """
    if not closes:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for sym, s in closes.items():
        if s is None:
            continue
        ser = pd.to_numeric(s, errors='coerce')
        if getattr(ser, 'index', None) is None:
            continue
        df = pd.DataFrame({sym: ser})
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(axis=0, how='any')
        df = df[~df.index.duplicated(keep='last')].sort_index()
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how='outer')

    out = out.sort_index()

    h = (how or 'inner').strip().lower()
    if h == 'inner':
        out = out.dropna(axis=0, how='any')
    else:
        out = out.ffill()
        out = out.dropna(axis=0, how='any')

    return out


def _log_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    if close_panel is None or close_panel.empty:
        return pd.DataFrame()
    x = close_panel.astype(float)
    return (np.log(x).diff(1)).dropna(axis=0, how='any')


def _write_symbol_charts(
    df: pd.DataFrame,
    *,
    symbol: str,
    period: str,
    interval: str,
    charts_dir: Path,
    forecast_steps: int,
) -> dict:
    """Write market-analyzer style artifacts (HTML + indicator CSV) for one symbol."""
    # Local imports so the hub can run without Plotly unless charts are requested.
    import plotly.graph_objects as go  # noqa: F401
    from dataclasses import asdict

    from utils.technical import (
        ichimoku,
        market_regime,
        rsi,
        support_resistance_from_pivots,
        trade_style_heuristic,
    )
    from utils.fourier import dominant_cycles_fft, rolling_dominant_period
    from projects.market_analyzer import build_candlestick_figure

    charts_dir.mkdir(parents=True, exist_ok=True)

    rsi14 = rsi(df['Close'], 14)
    ichi = ichimoku(df['High'], df['Low'], df['Close'])
    sr = support_resistance_from_pivots(df['High'], df['Low'], window=5)
    regime = market_regime(df['Close']).iloc[-1] if len(df) else 'neutral'
    style = trade_style_heuristic(df['Close'], df['High'], df['Low'], interval)
    fc = _arima_forecast(df['Close'], steps=forecast_steps)

    # Fourier / dominant cycles
    cycles = dominant_cycles_fft(df['Close'], window=min(256, len(df)), top_k=3, min_period=5.0, max_period=None)
    fft_roll = rolling_dominant_period(df['Close'], window=min(128, len(df)), min_period=5.0, max_period=None)

    # Indicator-enriched OHLCV for the charting subdir (kept separate from hub dataset CSV).
    out_df = df.copy()
    out_df['RSI14'] = rsi14
    out_df['Tenkan'] = ichi.tenkan_sen
    out_df['Kijun'] = ichi.kijun_sen
    out_df['SenkouA'] = ichi.senkou_span_a
    out_df['SenkouB'] = ichi.senkou_span_b

    out_df['FFTPeriod'] = fft_roll
    for ccy in cycles:
        out_df[f'FFTPeriod{ccy.k}'] = float(ccy.period)
        out_df[f'FFTPower{ccy.k}'] = float(ccy.power)
        out_df[f'FFTAmp{ccy.k}'] = float(ccy.amplitude)

    csv_path = charts_dir / f"{symbol}_{period}_{interval}.csv"
    out_df.to_csv(csv_path)

    fig = build_candlestick_figure(df, symbol, sr, ichi, rsi14)
    html_path = charts_dir / f"{symbol}_{period}_{interval}.html"
    fig.write_html(str(html_path))

    return {
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'last_close': float(df['Close'].iloc[-1]) if len(df) else None,
        'regime': str(regime),
        'support_levels': [float(x) for x in sr.supports],
        'resistance_levels': [float(x) for x in sr.resistances],
        'trade_style': asdict(style),
        'forecast_close_next': fc,
        'dominant_cycles': [
            {
                'k': int(c.k),
                'period': float(c.period),
                'frequency': float(c.frequency),
                'power': float(c.power),
                'amplitude': float(c.amplitude),
                'phase': float(c.phase),
            }
            for c in cycles
        ],
        'artifacts': {
            'csv': str(csv_path),
            'html': str(html_path),
        },
    }


def _parse_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _merge_on_date(df: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra is None or extra.empty:
        return df

    x = extra.copy()
    if "Date" in x.columns:
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
        x = x.dropna(subset=["Date"]).set_index("Date").sort_index()
    elif x.index.name is None:
        # Try first column as Date
        c0 = x.columns[0]
        x[c0] = pd.to_datetime(x[c0], errors="coerce")
        x = x.dropna(subset=[c0]).set_index(c0).sort_index()

    # align to df index; forward-fill macro across trading days
    x = x.reindex(df.index).ffill()
    merged = df.join(x, how="left")
    return merged


def _load_extra_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    return df


def _build_symbol_list(args) -> list[str]:
    symbols: list[str] = []

    if args.universe:
        u = args.universe.strip().lower()
        if u in {"sp500", "s&p500", "sp-500"}:
            symbols += fetch_sp500_symbols()
        elif u in {"nasdaq100", "nasdaq-100", "ndx"}:
            symbols += fetch_nasdaq100_symbols()
        else:
            raise SystemExit("Unknown universe. Use: sp500 | nasdaq100")

    symbols += [s.strip().upper() for s in _parse_list(args.symbols)]

    if args.symbols_file:
        symbols += load_symbols_file(args.symbols_file)

    # de-dupe, keep order
    seen = set()
    out = []
    for s in symbols:
        s = (s or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)

    if args.max_symbols is not None:
        out = out[: int(args.max_symbols)]

    if not out:
        raise SystemExit("No symbols specified. Use --symbols, --symbols-file, or --universe.")

    return out


def _build_macro_frames(args) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    fred_series = _parse_list(args.fred_series)
    if fred_series:
        try:
            api_key = args.fred_api_key or os.getenv("FRED_API_KEY")
            fred = fetch_fred_many(fred_series, start=args.fred_start, end=args.fred_end, api_key=api_key)
            if not fred.empty:
                frames.append(fred)
        except Exception as e:
            print(f"Warning: FRED fetch failed ({e}). Continuing without FRED.")

    # world bank syntax: INDICATOR@COUNTRY (country optional, default US)
    wb_specs = _parse_list(args.world_bank)
    if wb_specs:
        reqs: list[WorldBankRequest] = []
        for spec in wb_specs:
            if "@" in spec:
                ind, ctry = spec.split("@", 1)
                reqs.append(WorldBankRequest(indicator=ind.strip(), country=ctry.strip(), start=args.wb_start, end=args.wb_end))
            else:
                reqs.append(WorldBankRequest(indicator=spec.strip(), country="US", start=args.wb_start, end=args.wb_end))
        try:
            wb = fetch_world_bank_many(reqs)
            if not wb.empty:
                frames.append(wb)
        except Exception as e:
            print(f"Warning: World Bank fetch failed ({e}). Continuing without World Bank.")

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1).sort_index()
    # De-duplicate columns if repeated
    macro = macro.loc[:, ~macro.columns.duplicated()].copy()
    return macro


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a multi-source dataset and train")

    # universe / symbols
    p.add_argument("--universe", default=None, help="Universe: sp500 | nasdaq100")
    p.add_argument("--symbols", default=None, help="Comma-separated symbols (e.g., AAPL,MSFT,SPY,ETH-USD)")
    p.add_argument("--symbols-file", default=None, help="Path to .txt/.csv list of symbols")
    p.add_argument("--max-symbols", type=int, default=10, help="Limit how many symbols to process")

    # price data
    p.add_argument("--period", default="6mo", help="Price period (e.g., 6mo, 1y)")
    p.add_argument("--interval", default="1d", help="Price interval (e.g., 1d, 1h)")
    p.add_argument("--out-dir", default="hub_outputs", help="Where to write per-symbol CSVs")

    # output layout
    p.add_argument(
        "--assets-subdir",
        default=None,
        help=(
            "Optional subfolder under --out-dir for per-symbol datasets (e.g., 'assets'). "
            "Using a dedicated folder keeps output directories smaller and speeds up directory scans on Windows."
        ),
    )
    p.add_argument(
        "--shard-assets",
        action="store_true",
        help=(
            "When writing datasets into --assets-subdir, shard files into subfolders by first character of symbol. "
            "This avoids single directories with thousands of files (slow on Windows)."
        ),
    )

    # systemic-risk / microstructure monitoring
    p.add_argument(
        "--systemic-risk",
        action="store_true",
        help=(
            "After building datasets, run a systemic-risk pipeline: time×asset×feature tensor (CP/Tucker), "
            "2D embedding (t-SNE/Laplacian), rolling correlation networks + centrality, and ARIMA forecast of a stress index."
        ),
    )
    p.add_argument("--risk-out-dir", default=None, help="Override systemic-risk output folder (default: <out-dir>/systemic_risk)")
    p.add_argument("--tensor-method", default="cp", help="Systemic: tensor decomposition method (cp | tucker)")
    p.add_argument("--tensor-rank", type=int, default=4, help="Systemic: CP rank")
    p.add_argument("--tucker-ranks", default="4,4,3", help="Systemic: Tucker ranks (time,asset,feature) as i,j,k")
    p.add_argument("--tensor-features", default="returns,rv,depth", help="Systemic: comma-separated features for the tensor")
    p.add_argument("--rv-window", type=int, default=20, help="Systemic: realized volatility lookback window")
    p.add_argument("--depth-col", default=None, help="Systemic: optional column name for order-book depth")
    p.add_argument("--embed", default="tsne", help="Systemic: embedding method (tsne | laplacian)")
    p.add_argument("--corr-window", type=int, default=60, help="Systemic: rolling correlation window")
    p.add_argument("--corr-k", type=int, default=8, help="Systemic: kNN edges per node")
    p.add_argument("--centrality", default="pagerank", help="Systemic: pagerank | eigenvector | betweenness")
    p.add_argument("--arima-steps", type=int, default=5, help="Systemic: ARIMA forecast horizon")

    # optional chart/indicator artifacts (market-analyzer style)
    p.add_argument(
        "--generate-charts",
        action="store_true",
        help="Also write candlestick HTML + indicator CSVs per symbol (into <out-dir>/charts by default)",
    )
    p.add_argument(
        "--charts-dir",
        default=None,
        help="Override where chart artifacts are written (default: <out-dir>/charts)",
    )
    p.add_argument(
        "--forecast-steps",
        type=int,
        default=0,
        help="If >0, add a simple ARIMA forecast to chart summaries (default: 0/off)",
    )

    # optional chart selection
    p.add_argument(
        "--select-charts",
        action="store_true",
        help="After generating charts, score/select symbols meeting >= --min-conditions and copy them into selected_charts/",
    )
    p.add_argument("--min-conditions", type=int, default=3, help="Minimum conditions (out of 4) required to select")
    p.add_argument("--max-selected", type=int, default=50, help="Max charts to select (after filtering)")
    p.add_argument("--news-days", type=int, default=3, help="News lookback window (days)")
    p.add_argument("--news-min-items", type=int, default=1, help="Minimum recent RSS items to consider 'in the news'")
    p.add_argument("--skip-news", action="store_true", help="Disable the news condition (condition #3)")
    p.add_argument("--fib-lookback", type=int, default=60, help="Fib swing lookback window (candles)")
    p.add_argument("--fib-tolerance", type=float, default=0.01, help="Relative tolerance for fib proximity (e.g. 0.01 = 1%)")

    # spectral graph / DMD analysis (multi-asset)
    p.add_argument("--spectral", action="store_true", help="Compute similarity graph + Fiedler clustering across assets")
    p.add_argument("--spectral-k", type=int, default=8, help="kNN neighbors for the similarity graph")
    p.add_argument("--spectral-abs", action="store_true", help="Use absolute correlation for similarity")
    p.add_argument("--spectral-label", default='median', help="Fiedler bipartition rule: median | sign")
    p.add_argument("--spectral-min-overlap", type=int, default=80, help="Minimum overlapping return rows required")

    p.add_argument("--dmd", action="store_true", help="Run Dynamic Mode Decomposition on aligned multi-asset returns")
    p.add_argument("--dmd-rank", type=int, default=6, help="DMD rank (SVD truncation)")

    # portfolio allocation (analysis-only; outputs target weights)
    p.add_argument(
        "--allocate-portfolio",
        action="store_true",
        help="Compute target portfolio weights (optionally cluster-aware) and write them into <out-dir>/",
    )
    p.add_argument(
        "--alloc-universe",
        default="all",
        help="Which symbols to allocate over: all | selected (requires --select-charts)",
    )
    p.add_argument("--alloc-lookback", type=int, default=60, help="Lookback window (rows) for volatility estimates")
    p.add_argument("--alloc-min-overlap", type=int, default=80, help="Minimum overlapping return rows required")
    p.add_argument(
        "--alloc-cluster-budget",
        default="inverse_vol",
        help="How to allocate across clusters: inverse_vol | equal",
    )
    p.add_argument(
        "--alloc-within-cluster",
        default="inverse_vol",
        help="How to allocate within a cluster: inverse_vol | equal",
    )
    p.add_argument("--alloc-max-weight", type=float, default=None, help="Optional per-asset max weight cap")
    p.add_argument("--alloc-min-weight", type=float, default=0.0, help="Optional per-asset min weight floor")

    # macro
    p.add_argument("--fred-series", default=None, help="Comma-separated FRED series ids (e.g., CPIAUCSL,UNRATE)")
    p.add_argument("--fred-api-key", default=None, help="FRED API key (or set FRED_API_KEY env)")
    p.add_argument("--fred-start", default=None, help="FRED observation_start (YYYY-MM-DD)")
    p.add_argument("--fred-end", default=None, help="FRED observation_end (YYYY-MM-DD)")

    p.add_argument("--world-bank", default=None, help="Comma-separated World Bank indicators, optional @COUNTRY (e.g., FP.CPI.TOTL.ZG@US)")
    p.add_argument("--wb-start", default=None, help="World Bank start year (YYYY)")
    p.add_argument("--wb-end", default=None, help="World Bank end year (YYYY)")

    # local exogenous CSVs (Kaggle/AEA/etc.): merged by Date
    p.add_argument("--extra-csv", default=None, help="Path to a local CSV with a Date column to join as features")

    # training options (forwarded to ml_train)
    p.add_argument("--model", default="hgb")
    p.add_argument("--task", default="classification")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.003)
    p.add_argument("--cv", default="walkforward")
    p.add_argument("--n-splits", type=int, default=6)
    p.add_argument("--test-window", type=int, default=20)
    p.add_argument("--purge", type=int, default=None)
    p.add_argument("--importance", default="permutation")
    p.add_argument("--top-features", type=int, default=25)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--ml-out-dir", default=None, help="Where to write ML outputs (default: <out-dir>_ml)")
    p.add_argument("--skip-train", action="store_true", help="Only build datasets; do not train")

    args = p.parse_args(argv)

    symbols = _build_symbol_list(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets_dir: Path | None = None
    if args.assets_subdir:
        assets_dir = out_dir / str(args.assets_subdir)
        assets_dir.mkdir(parents=True, exist_ok=True)

    charts_dir = Path(args.charts_dir) if args.charts_dir else (out_dir / "charts")

    macro = _build_macro_frames(args)
    extra = _load_extra_csv(args.extra_csv) if args.extra_csv else pd.DataFrame()

    # Cache macro frames for inspection
    if not macro.empty:
        macro.to_csv(out_dir / "macro_features.csv")

    if args.extra_csv:
        # write a copy to keep provenance near outputs
        try:
            extra.to_csv(out_dir / "extra_features_raw.csv", index=False)
        except Exception:
            pass

    failures: list[dict] = []
    results: list[dict] = []

    chart_failures: list[dict] = []
    chart_summaries: list[dict] = []

    scored_for_selection: list[dict] = []
    _news_cache: dict[str, tuple[bool, int]] = {}

    selected_symbols: list[str] = []
    selection_manifest_path: str | None = None

    # Collect closes for multi-asset analyses (spectral/DMD)
    closes_by_symbol: dict[str, pd.Series] = {}

    for sym in symbols:
        try:
            df = fetch_ohlcv(OHLCVRequest(symbol=sym, period=args.period, interval=args.interval))
        except Exception as e:
            failures.append({"symbol": sym, "error": str(e)})
            print(f"Warning: skipping {sym} due to data error: {e}")
            continue

        df_out = df.copy()
        if not macro.empty:
            df_out = _merge_on_date(df_out, macro)
        if args.extra_csv and not extra.empty:
            df_out = _merge_on_date(df_out, extra)

        csv_name = f"{sym}_{args.period}_{args.interval}.csv"
        if assets_dir is None:
            csv_path = out_dir / csv_name
        else:
            if args.shard_assets:
                # Shard by first visible character to keep each directory small.
                # Symbols like 'ETH-USD' -> 'E', 'EURUSD=X' -> 'E', '^GSPC' -> '^'.
                shard = (sym or "_")[:1]
                shard_dir = assets_dir / shard
                shard_dir.mkdir(parents=True, exist_ok=True)
                csv_path = shard_dir / csv_name
            else:
                csv_path = assets_dir / csv_name
        df_out.to_csv(csv_path)
        print(f"Wrote dataset: {csv_path}")

        # Store close series for multi-asset analysis
        try:
            closes_by_symbol[sym] = df['Close'].copy()
        except Exception:
            pass
        result_entry = {
            "symbol": sym,
            "rows": int(len(df_out)),
            "columns": int(df_out.shape[1]),
            "csv": str(csv_path),
        }

        if args.generate_charts:
            try:
                chart_summary = _write_symbol_charts(
                    df,
                    symbol=sym,
                    period=args.period,
                    interval=args.interval,
                    charts_dir=charts_dir,
                    forecast_steps=int(args.forecast_steps or 0),
                )
                chart_summaries.append(chart_summary)
                result_entry["chart"] = chart_summary.get("artifacts")

                # Condition scoring for selection (requires charts to exist)
                if args.select_charts:
                    ichi_ok, direction, ichi_dbg = _ichimoku_trade_signal(df)
                    rsi_ok, rsi_val = _rsi_trade_ok(df, direction=direction)
                    fib_ok, fib_levels, fib_nearest = _fib_check(
                        df,
                        direction=direction or 'long',
                        lookback=int(args.fib_lookback),
                        tolerance=float(args.fib_tolerance),
                    )

                    news_ok = False
                    news_count = 0
                    if not args.skip_news and _is_equity_symbol(sym):
                        if sym not in _news_cache:
                            _news_cache[sym] = _news_hit(
                                sym,
                                days=int(args.news_days),
                                min_items=int(args.news_min_items),
                            )
                        news_ok, news_count = _news_cache[sym]

                    conds = {
                        'ichimoku': bool(ichi_ok),
                        'rsi': bool(rsi_ok),
                        'news': bool(news_ok) if not args.skip_news else None,
                        'fibonacci': bool(fib_ok),
                    }
                    conds_met = sum(1 for v in conds.values() if v is True)
                    score = float(conds_met) / 4.0

                    scored_for_selection.append({
                        'symbol': sym,
                        'direction': direction,
                        'conditions': conds,
                        'conditions_met': int(conds_met),
                        'score': score,
                        'rsi14': rsi_val,
                        'news_recent_items': int(news_count),
                        'fib_nearest': fib_nearest,
                        'fib_levels': fib_levels,
                        'ichimoku_debug': ichi_dbg,
                        'artifacts': chart_summary.get('artifacts') or {},
                    })
            except Exception as e:
                chart_failures.append({"symbol": sym, "error": str(e)})
                print(f"Warning: chart generation failed for {sym}: {e}")

        results.append(result_entry)

    if failures:
        print(f"Note: {len(failures)} symbols failed. Continuing to training with the remaining.")

    if args.generate_charts:
        # Keep a market-analyzer-like summary near the chart artifacts.
        charts_dir.mkdir(parents=True, exist_ok=True)
        with open(charts_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"results": chart_summaries, "failures": chart_failures}, f, indent=2)

        if args.select_charts:
            sel = _select_and_copy_charts(
                out_dir=out_dir,
                charts_dir=charts_dir,
                scored=scored_for_selection,
                min_conditions=int(args.min_conditions),
                max_selected=int(args.max_selected) if args.max_selected is not None else None,
            )
            selection_manifest_path = str(out_dir / 'selected_charts.json')
            try:
                selected_symbols = [str(x.get('symbol')).strip().upper() for x in (sel.get('selected') or []) if x.get('symbol')]
            except Exception:
                selected_symbols = []
            print(f"Wrote selection manifest: {out_dir / 'selected_charts.json'}")
            print(f"Selected charts copied to: {out_dir / 'selected_charts'}")

    # --- Multi-asset analysis: spectral clustering / DMD / portfolio allocation ---
    spectral_clusters_csv: str | None = None
    spectral_graph_json: str | None = None
    dmd_eigs_csv: str | None = None
    dmd_summary_json: str | None = None
    portfolio_weights_csv: str | None = None
    portfolio_allocation_json: str | None = None

    systemic_out_dir: str | None = None
    systemic_artifacts: dict | None = None

    spec_labels_by_symbol: dict[str, int] | None = None

    if args.spectral or args.dmd or args.allocate_portfolio:
        panel = _align_closes(closes_by_symbol, how='inner')
        rets = _log_returns(panel)

        # Require a minimum overlap so results aren't junk.
        enough_for_spectral_dmd = (not rets.empty) and (rets.shape[0] >= int(args.spectral_min_overlap)) and (rets.shape[1] >= 3)
        if enough_for_spectral_dmd:
            if args.spectral:
                try:
                    spec = spectral_cluster_assets(
                        rets,
                        k=int(args.spectral_k),
                        normalized=True,
                        corr_method='pearson',
                        use_abs_corr=bool(args.spectral_abs),
                        label_method=str(args.spectral_label),
                    )

                    if spec.labels is not None and spec.fiedler_vector is not None:
                        dfc = pd.DataFrame({
                            'symbol': spec.symbols,
                            'cluster': spec.labels.astype(int),
                            'fiedler': spec.fiedler_vector.astype(float),
                        }).sort_values(['cluster', 'symbol'])

                        out_csv = out_dir / 'spectral_clusters.csv'
                        dfc.to_csv(out_csv, index=False)

                        spec_labels_by_symbol = {str(r['symbol']): int(r['cluster']) for _, r in dfc.iterrows()}

                        out_json = out_dir / 'spectral_graph.json'
                        with open(out_json, 'w', encoding='utf-8') as f:
                            json.dump({
                                'symbols': spec.symbols,
                                'params': {
                                    'k': int(args.spectral_k),
                                    'abs_corr': bool(args.spectral_abs),
                                    'label_method': str(args.spectral_label),
                                    'min_overlap': int(args.spectral_min_overlap),
                                },
                                'fiedler_value': spec.fiedler_value,
                                'eigenvalues': [float(x) for x in (spec.eigenvalues[: min(50, spec.eigenvalues.size)]).tolist()],
                                'rows_used': int(rets.shape[0]),
                                'assets_used': int(rets.shape[1]),
                            }, f, indent=2)

                        spectral_clusters_csv = str(out_csv)
                        spectral_graph_json = str(out_json)

                        print(f"Wrote spectral clusters: {out_csv}")
                except Exception as e:
                    print(f"Warning: spectral clustering failed: {e}")

            if args.dmd:
                try:
                    # DMD expects (n_features, n_time)
                    X = rets.to_numpy(dtype=float).T
                    res = exact_dmd(X, rank=int(args.dmd_rank))
                    growth, freq = eigenvalue_to_frequency(res.eigenvalues, dt=1.0)

                    out_eigs = out_dir / 'dmd_eigs.csv'
                    df_e = pd.DataFrame({
                        'eig_real': np.real(res.eigenvalues),
                        'eig_imag': np.imag(res.eigenvalues),
                        'growth': growth,
                        'frequency': freq,
                        'amplitude_abs': np.abs(res.amplitudes),
                    }).sort_values('amplitude_abs', ascending=False)
                    df_e.to_csv(out_eigs, index=False)

                    out_json = out_dir / 'dmd_summary.json'
                    with open(out_json, 'w', encoding='utf-8') as f:
                        json.dump({
                            'rank': int(res.rank),
                            'rows_used': int(rets.shape[0]),
                            'assets_used': int(rets.shape[1]),
                            'assets': list(rets.columns),
                            'top_modes': int(min(10, df_e.shape[0])),
                        }, f, indent=2)

                    dmd_eigs_csv = str(out_eigs)
                    dmd_summary_json = str(out_json)

                    print(f"Wrote DMD eigs: {out_eigs}")
                except Exception as e:
                    print(f"Warning: DMD failed: {e}")

        else:
            if args.spectral or args.dmd:
                print(
                    "Note: skipping spectral/DMD (need at least 3 assets and enough overlapping rows). "
                    f"Have rows={rets.shape[0] if rets is not None else 0}, assets={rets.shape[1] if rets is not None else 0}."
                )

        # Portfolio allocation can run with >= 2 assets.
        if args.allocate_portfolio:
            min_rows = int(args.alloc_min_overlap)
            if not rets.empty and rets.shape[0] >= min_rows and rets.shape[1] >= 2:
                alloc_universe = (args.alloc_universe or 'all').strip().lower()
                if alloc_universe in {'selected', 'select', 'signals'} and selected_symbols:
                    alloc_syms = [s for s in selected_symbols if s in set(rets.columns)]
                else:
                    alloc_syms = list(rets.columns)

                if len(alloc_syms) >= 2:
                    rets_alloc = rets[alloc_syms].copy()
                    # If we have spectral cluster labels from this run, use them; else put everything in one cluster.
                    clusters_map = None
                    if spec_labels_by_symbol is not None:
                        clusters_map = {s: int(spec_labels_by_symbol.get(s, 0)) for s in alloc_syms}

                    alloc = cluster_inverse_vol_allocation(
                        rets_alloc,
                        clusters=clusters_map,
                        lookback=int(args.alloc_lookback),
                        cluster_budget=str(args.alloc_cluster_budget),
                        within_cluster=str(args.alloc_within_cluster),
                        min_weight=float(args.alloc_min_weight or 0.0),
                        max_weight=float(args.alloc_max_weight) if args.alloc_max_weight is not None else None,
                    )

                    # Write CSV + JSON artifacts
                    out_w = out_dir / 'portfolio_weights.csv'
                    out_j = out_dir / 'portfolio_allocation.json'

                    diag = alloc.diagnostics or {}
                    asset_vol = diag.get('asset_vol') or {}
                    rc = diag.get('risk_contrib_diag') or {}
                    cl_map = diag.get('clusters') or {}

                    dfw = pd.DataFrame({
                        'symbol': alloc.weights.index,
                        'weight': [float(alloc.weights.loc[s]) for s in alloc.weights.index],
                        'cluster': [int(cl_map.get(s, 0)) for s in alloc.weights.index],
                        'vol': [float(asset_vol.get(s, float('nan'))) for s in alloc.weights.index],
                        'risk_contrib_diag': [float(rc.get(s, float('nan'))) for s in alloc.weights.index],
                    }).sort_values(['cluster', 'weight'], ascending=[True, False])

                    dfw.to_csv(out_w, index=False)
                    with open(out_j, 'w', encoding='utf-8') as f:
                        json.dump({
                            'universe': alloc_universe,
                            'symbols_used': list(alloc.weights.index),
                            'weights': {k: float(v) for k, v in alloc.weights.to_dict().items()},
                            'diagnostics': diag,
                        }, f, indent=2)

                    portfolio_weights_csv = str(out_w)
                    portfolio_allocation_json = str(out_j)
                    print(f"Wrote portfolio weights: {out_w}")
                else:
                    print("Note: skipping portfolio allocation (need at least 2 selected assets with enough overlap).")
            else:
                print(
                    "Note: skipping portfolio allocation (need at least 2 assets and enough overlapping rows). "
                    f"Have rows={rets.shape[0] if rets is not None else 0}, assets={rets.shape[1] if rets is not None else 0}."
                )

    # --- Systemic-risk monitoring (tensor + networks + ARIMA) ---
    if args.systemic_risk:
        try:
            from utils.systemic_risk import run_systemic_risk

            tucker_ranks = tuple(int(x.strip()) for x in str(args.tucker_ranks).split(',') if x.strip())
            if len(tucker_ranks) != 3:
                raise ValueError("--tucker-ranks must be 3 integers like 4,4,3")

            feats = [x.strip() for x in str(args.tensor_features).split(',') if x.strip()]

            res = run_systemic_risk(
                hub_dir=out_dir,
                out_dir=args.risk_out_dir,
                assets_subdir=str(args.assets_subdir) if args.assets_subdir else None,
                tensor_method=str(args.tensor_method),
                tensor_rank=int(args.tensor_rank),
                tucker_ranks=tucker_ranks,  # type: ignore[arg-type]
                embed_method=str(args.embed),
                features=feats,
                vol_window=int(args.rv_window),
                depth_col=args.depth_col,
                corr_window=int(args.corr_window),
                corr_k=int(args.corr_k),
                centrality=str(args.centrality),
                arima_steps=int(args.arima_steps),
                random_state=int(args.random_state),
            )

            systemic_out_dir = str(res.get('out_dir')) if res else None
            systemic_artifacts = (res.get('artifacts') if isinstance(res, dict) else None)  # type: ignore[assignment]
            print(f"Wrote systemic-risk artifacts: {systemic_out_dir}")
        except Exception as e:
            print(f"Warning: systemic-risk monitoring failed: {e}")

    # Write a run manifest for provenance (after optional analyses so it can reference all artifacts)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "universe": args.universe,
            "symbols_requested": symbols,
            "period": args.period,
            "interval": args.interval,
            "results": results,
            "failures": failures,
            "charts": {
                "enabled": bool(args.generate_charts),
                "charts_dir": str(charts_dir) if args.generate_charts else None,
                "summary_json": str(charts_dir / "summary.json") if args.generate_charts else None,
                "selection_manifest": selection_manifest_path,
                "selected_symbols": selected_symbols,
                "failures": chart_failures if args.generate_charts else [],
            },
            "spectral": {
                "enabled": bool(args.spectral),
                "clusters_csv": spectral_clusters_csv,
                "graph_json": spectral_graph_json,
            },
            "dmd": {
                "enabled": bool(args.dmd),
                "eigs_csv": dmd_eigs_csv,
                "summary_json": dmd_summary_json,
            },
            "portfolio": {
                "enabled": bool(args.allocate_portfolio),
                "universe": (args.alloc_universe or 'all'),
                "weights_csv": portfolio_weights_csv,
                "allocation_json": portfolio_allocation_json,
            },
            "systemic_risk": {
                "enabled": bool(args.systemic_risk),
                "out_dir": systemic_out_dir,
                "artifacts": systemic_artifacts or {},
            },
            "macro_features_csv": str(out_dir / "macro_features.csv") if (out_dir / "macro_features.csv").exists() else None,
            "extra_features_raw_csv": str(out_dir / "extra_features_raw.csv") if (out_dir / "extra_features_raw.csv").exists() else None,
        }, f, indent=2)

    if args.skip_train:
        # Dataset-only mode
        # If price data succeeded but charting failed, still return non-zero so it is noticeable.
        if failures:
            return 2
        if args.generate_charts and chart_failures:
            return 2
        return 0

    # Train using existing ml_train, multi-asset mode
    from projects.ml_train import main as ml_main

    ml_out = Path(args.ml_out_dir) if args.ml_out_dir else Path(str(out_dir) + "_ml")
    purge = int(args.purge) if args.purge is not None else int(args.horizon)

    ml_args = [
        "--in-dir",
        str(out_dir),
        "--multi-asset",
        "--model",
        str(args.model),
        "--task",
        str(args.task),
        "--horizon",
        str(args.horizon),
        "--threshold",
        str(args.threshold),
        "--cv",
        str(args.cv),
        "--n-splits",
        str(args.n_splits),
        "--test-window",
        str(args.test_window),
        "--purge",
        str(purge),
        "--importance",
        str(args.importance),
        "--top-features",
        str(args.top_features),
        "--random-state",
        str(args.random_state),
        "--out-dir",
        str(ml_out),
    ]

    try:
        rc = ml_main(ml_args)
        return int(rc)
    except Exception as e:
        # Keep the dataset artifacts even if training can't run (e.g., too few rows for 1d/1w).
        ml_out.mkdir(parents=True, exist_ok=True)
        with open(ml_out / "train_status.json", "w", encoding="utf-8") as f:
            json.dump({"status": "failed", "error": str(e), "ml_args": ml_args}, f, indent=2)
        print(f"Warning: training failed, wrote {ml_out / 'train_status.json'}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
