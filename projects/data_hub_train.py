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
from pathlib import Path

import pandas as pd

from utils.market_data import OHLCVRequest, fetch_ohlcv
from utils.macro_data import FREDSeriesRequest, WorldBankRequest, fetch_fred_many, fetch_world_bank_many
from utils.universe import fetch_nasdaq100_symbols, fetch_sp500_symbols, load_symbols_file


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
    from projects.market_analyzer import build_candlestick_figure

    charts_dir.mkdir(parents=True, exist_ok=True)

    rsi14 = rsi(df['Close'], 14)
    ichi = ichimoku(df['High'], df['Low'], df['Close'])
    sr = support_resistance_from_pivots(df['High'], df['Low'], window=5)
    regime = market_regime(df['Close']).iloc[-1] if len(df) else 'neutral'
    style = trade_style_heuristic(df['Close'], df['High'], df['Low'], interval)
    fc = _arima_forecast(df['Close'], steps=forecast_steps)

    # Indicator-enriched OHLCV for the charting subdir (kept separate from hub dataset CSV).
    out_df = df.copy()
    out_df['RSI14'] = rsi14
    out_df['Tenkan'] = ichi.tenkan_sen
    out_df['Kijun'] = ichi.kijun_sen
    out_df['SenkouA'] = ichi.senkou_span_a
    out_df['SenkouB'] = ichi.senkou_span_b

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

        csv_path = out_dir / f"{sym}_{args.period}_{args.interval}.csv"
        df_out.to_csv(csv_path)
        print(f"Wrote dataset: {csv_path}")
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
            except Exception as e:
                chart_failures.append({"symbol": sym, "error": str(e)})
                print(f"Warning: chart generation failed for {sym}: {e}")

        results.append(result_entry)

    if failures:
        print(f"Note: {len(failures)} symbols failed. Continuing to training with the remaining.")

    # Write a run manifest for provenance
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
                "failures": chart_failures if args.generate_charts else [],
            },
            "macro_features_csv": str(out_dir / "macro_features.csv") if (out_dir / "macro_features.csv").exists() else None,
            "extra_features_raw_csv": str(out_dir / "extra_features_raw.csv") if (out_dir / "extra_features_raw.csv").exists() else None,
        }, f, indent=2)

    if args.generate_charts:
        # Keep a market-analyzer-like summary near the chart artifacts.
        charts_dir.mkdir(parents=True, exist_ok=True)
        with open(charts_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"results": chart_summaries, "failures": chart_failures}, f, indent=2)

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
