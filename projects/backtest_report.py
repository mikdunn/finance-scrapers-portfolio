"""Backtest comparison report.

Scans backtest output folders (each containing backtest_summary.json and trades.csv)
and produces:
- strategy_comparison.csv: aggregated metrics per run
- strategy_report.html: Plotly charts showing how different "components" (stops,
  trailing stops, vol targeting, proba signals, etc.) relate to performance.

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.backtest import compute_metrics, compute_trade_metrics


def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_curve(path: Path) -> pd.DataFrame:
    """Read an equity curve CSV written by this repo.

    Handles both styles:
      - simulate_ohlc: index is 'time'
      - compute_equity_curve: datetime index persisted as first unnamed column
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # Prefer explicit time column.
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()
        return df

    # Common case: first column is the saved index.
    if df.columns.size and str(df.columns[0]).lower().startswith("unnamed"):
        idx = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.drop(columns=[df.columns[0]])
        df.index = idx
        df = df[~df.index.isna()].sort_index()
        return df

    # Fallback: look for Date/Datetime columns.
    for c in ("Date", "Datetime", "date", "datetime"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).set_index(c).sort_index()
            return df

    return df


def _pretty_metric_name(col: str) -> str:
    m = {
        "m_sharpe": "Sharpe ratio (annualized)",
        "m_total_return": "Total return",
        "m_cagr": "CAGR",
        "m_max_drawdown": "Max drawdown",
        "t_win_rate": "Win rate",
        "t_profit_factor": "Profit factor",
        "t_expectancy": "Avg trade return (expectancy)",
    }
    return m.get(str(col), str(col))
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _discover_runs(root: Path, pattern: str) -> list[Path]:
    runs: list[Path] = []
    for p in root.glob(pattern):
        if p.is_dir() and (p / "backtest_summary.json").exists():
            runs.append(p)
    return sorted(runs)


_CRYPTO_BASES = {
    # Common Yahoo crypto bases
    "BTC",
    "ETH",
    "SOL",
    "DOGE",
    "ADA",
    "XRP",
    "BNB",
    "AVAX",
    "DOT",
    "MATIC",
    "LTC",
    "LINK",
    "BCH",
    "ATOM",
    "TRX",
    "XLM",
    "UNI",
    "AAVE",
    "SHIB",
    "USDT",
    "USDC",
}


def _is_crypto_symbol(symbol: str | None) -> bool:
    if symbol is None:
        return False
    s = str(symbol).strip().upper()
    if not s:
        return False

    # Yahoo Finance crypto usually looks like "BTC-USD".
    m = re.match(r"^([A-Z0-9]{2,10})-USD$", s)
    if m:
        base = m.group(1)
        return base in _CRYPTO_BASES

    # Also treat explicit coin tickers as crypto when they appear without suffix.
    return s in _CRYPTO_BASES


def _run_has_crypto(run_dir: Path) -> bool:
    """Return True if this run appears to include crypto assets."""
    try:
        s = _load_json(run_dir / "backtest_summary.json")
    except Exception:
        s = {}

    # Single-asset runs generally have an 'asset' field.
    if _is_crypto_symbol(s.get("asset")):
        return True

    # Portfolio runs: prefer assets_metrics.csv if present.
    am = run_dir / "assets_metrics.csv"
    if am.exists():
        try:
            df = pd.read_csv(am)
            if "asset" in df.columns:
                for a in df["asset"].dropna().astype(str).tolist():
                    if _is_crypto_symbol(a):
                        return True
        except Exception:
            pass

    # Portfolio runs may have per-asset subfolders under /assets.
    assets_dir = run_dir / "assets"
    if assets_dir.exists():
        try:
            for p in assets_dir.iterdir():
                if p.is_dir() and _is_crypto_symbol(p.name):
                    return True
        except Exception:
            pass

    return False


def _should_exclude_run_for_crypto(run_dir: Path) -> bool:
    """Return True if this run should be excluded when --exclude-crypto is set.

    Rules:
    - Single-asset runs: exclude if the asset looks like crypto.
    - Portfolio runs: exclude only if we can determine that *all* assets are crypto.
      (If a portfolio run contains any non-crypto asset, keep it.)
    """
    try:
        s = _load_json(run_dir / "backtest_summary.json")
    except Exception:
        s = {}

    is_portfolio = "portfolio_metrics" in (s or {})
    if not is_portfolio:
        return _is_crypto_symbol(s.get("asset"))

    # Portfolio run: inspect assets if we can.
    assets: list[str] = []
    am = run_dir / "assets_metrics.csv"
    if am.exists():
        try:
            df = pd.read_csv(am)
            if "asset" in df.columns:
                assets = [str(x) for x in df["asset"].dropna().tolist()]
        except Exception:
            assets = []

    if not assets:
        assets_dir = run_dir / "assets"
        if assets_dir.exists():
            try:
                assets = [p.name for p in assets_dir.iterdir() if p.is_dir()]
            except Exception:
                assets = []

    # If we couldn't detect assets, keep the run (fail-open).
    if not assets:
        return False

    has_non_crypto = any(not _is_crypto_symbol(a) for a in assets)
    has_crypto = any(_is_crypto_symbol(a) for a in assets)
    return bool(has_crypto and not has_non_crypto)


def _components_from_summary(s: dict) -> dict:
    mode = str(s.get("mode", "")).strip().lower()
    signal_source = str(s.get("signal_source", "")).strip().lower()

    stop_loss = s.get("stop_loss")
    take_profit = s.get("take_profit")
    trailing_stop = s.get("trailing_stop")
    trailing_atr_mult = s.get("trailing_atr_mult")
    vol_target = s.get("vol_target")

    def _is_set(v) -> bool:
        try:
            return v is not None and float(v) > 0
        except Exception:
            return v is not None

    # Extra experimental components may be present in some runs (e.g., strategy sweeps).
    def _boolish(v) -> bool:
        try:
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(v)
        except Exception:
            return False

    return {
        "long_only": mode in {"long", "long_only", "long-only"},
        "uses_proba_signal": signal_source in {"proba", "prob", "predict_proba"},
        "uses_stop_loss": _is_set(stop_loss),
        "uses_take_profit": _is_set(take_profit),
        "uses_trailing_stop": _is_set(trailing_stop),
        "uses_trailing_atr": _is_set(trailing_atr_mult),
        "uses_vol_target": _is_set(vol_target),
        # Experimental / advanced components
        "uses_hmm": _boolish(s.get("uses_hmm")),
        "uses_fft": _boolish(s.get("uses_fft")),
        "uses_wavelet": _boolish(s.get("uses_wavelet")),
        "uses_dmd": _boolish(s.get("uses_dmd")),
        "uses_neural_net": _boolish(s.get("uses_neural_net"))
        or str(s.get("model", "")).strip().lower() in {"mlp", "nn", "neural", "neural_net"},
    }


def _flatten_metrics(prefix: str, d: dict | None) -> dict:
    out: dict = {}
    for k, v in (d or {}).items():
        if isinstance(v, (dict, list)):
            continue
        out[f"{prefix}{k}"] = v
    return out


def _combo_from_components(d: dict) -> str:
    """Build a stable, human-readable combo string from component flags."""
    model = None
    for k in ("model", "core_model"):
        if d.get(k) is not None:
            model = str(d.get(k)).strip().lower()
            break
    model_label = None
    if model:
        # keep short / stable labels
        m = {
            "histgradientboosting": "hgb",
            "histgradientboostingclassifier": "hgb",
            "randomforest": "rf",
            "randomforestclassifier": "rf",
            "mlpclassifier": "mlp",
            "neural_net": "mlp",
            "neural": "mlp",
            "nn": "mlp",
        }
        model_label = m.get(model, model)

    # If a run records a specific wavelet kernel (strategy sweep), use it in the label.
    wk = d.get("wavelet_kernel")
    wp = d.get("wavelet_ar_p")
    wavelet_label = "wavelet"
    if wk is not None:
        s = str(wk).strip().lower()
        if s.startswith("dwt_db"):
            wavelet_label = "wv_" + s.replace("dwt_", "")
        elif "morlet" in s:
            wavelet_label = "wv_morlet"
        elif "ricker" in s or "mexh" in s:
            wavelet_label = "wv_ricker"
        else:
            wavelet_label = "wv_" + re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if wp is not None:
        try:
            wavelet_label = f"{wavelet_label}_ar{int(wp)}"
        except Exception:
            pass

    order = [
        ("long_only", "long_only"),
        ("uses_proba_signal", "proba"),
        ("uses_hmm", "hmm"),
        ("uses_fft", "fft"),
        ("uses_wavelet", wavelet_label),
        ("uses_dmd", "dmd"),
        ("uses_neural_net", "nn"),
        ("uses_stop_loss", "stop"),
        ("uses_take_profit", "tp"),
        ("uses_trailing_stop", "trail"),
        ("uses_trailing_atr", "trail_atr"),
        ("uses_vol_target", "vol_target"),
    ]
    parts: list[str] = []
    if model_label:
        parts.append(model_label)
    for k, label in order:
        try:
            if bool(d.get(k)):
                parts.append(label)
        except Exception:
            continue
    return "+".join(parts) if parts else (model_label or "baseline")


def _merge_asset_metrics(
    run_rows_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    *,
    asset: str,
) -> pd.DataFrame:
    """Return one row per run with metrics for a single asset.

    Keeps run-level configuration/components, replaces m_* metrics with per-asset
    m_* columns where available.
    """
    if run_rows_df is None or run_rows_df.empty:
        return pd.DataFrame()
    if assets_df is None or assets_df.empty or "asset" not in assets_df.columns:
        return pd.DataFrame()

    a = str(asset)
    adf = assets_df[assets_df["asset"].astype(str) == a].copy()
    if adf.empty:
        return pd.DataFrame()

    # Keep latest per run if duplicates exist.
    adf = adf.dropna(subset=["run"]).drop_duplicates(subset=["run"], keep="last")

    # Separate config columns vs metric columns.
    metric_cols = [c for c in adf.columns if str(c).startswith("m_")]
    keep_cols = [
        c
        for c in run_rows_df.columns
        if (not str(c).startswith("m_")) and (c not in metric_cols)
    ]

    # Avoid duplicate columns (run_dir/asset) after merge.
    keep_cols = [c for c in keep_cols if c not in {"run_dir", "asset"}]

    merged = run_rows_df[keep_cols].merge(adf[["run", "run_dir", "asset"] + metric_cols], on=["run"], how="inner")
    return merged


def _build_report_rows(run_dir: Path, *, recompute_metrics: bool) -> dict:
    s = _load_json(run_dir / "backtest_summary.json")

    # Determine whether this is a portfolio (multi-asset) run.
    is_portfolio = "portfolio_metrics" in s

    # Trade metrics: prefer those already in summary (single-asset new runs),
    # else compute from trades.csv where available.
    trade_metrics = s.get("trade_metrics") if isinstance(s.get("trade_metrics"), dict) else None
    if trade_metrics is None and not is_portfolio:
        trade_metrics = compute_trade_metrics(_read_trades(run_dir / "trades.csv"))

    # For portfolio runs, compute an "average" trade metric across per-asset trades if present.
    if trade_metrics is None and is_portfolio:
        assets_dir = run_dir / "assets"
        if assets_dir.exists():
            tms: list[dict] = []
            for asset in assets_dir.iterdir():
                if not asset.is_dir():
                    continue
                tm = compute_trade_metrics(_read_trades(asset / "trades.csv"))
                if int(tm.get("n_trades", 0) or 0) > 0:
                    tms.append(tm)
            if tms:
                # simple average of rates and factors; weighted averaging could be added later
                trade_metrics = {
                    "n_trades": int(sum(int(x.get("n_trades", 0) or 0) for x in tms)),
                    "win_rate": float(np.nanmean([_safe_float(x.get("win_rate")) for x in tms])),
                    "profit_factor": float(np.nanmean([_safe_float(x.get("profit_factor")) for x in tms])),
                }

    metrics = s.get("metrics") if isinstance(s.get("metrics"), dict) else {}
    portfolio_metrics = s.get("portfolio_metrics") if isinstance(s.get("portfolio_metrics"), dict) else {}

    # Optionally recompute curve metrics so changes in annualization/Sharpe are reflected.
    if recompute_metrics:
        if is_portfolio:
            curve_path = run_dir / "portfolio_equity_curve.csv"
        else:
            curve_path = run_dir / "equity_curve.csv"
        curve = _read_curve(curve_path)
        if not curve.empty:
            try:
                cm = compute_metrics(curve)
                if is_portfolio:
                    portfolio_metrics = cm
                else:
                    metrics = cm
            except Exception:
                pass

    base = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "is_portfolio": bool(is_portfolio),
        "asset": s.get("asset"),
        "mode": s.get("mode"),
        "task": s.get("task"),
        "signal_source": s.get("signal_source"),
        "price": s.get("price"),
        "stop_loss": s.get("stop_loss"),
        "take_profit": s.get("take_profit"),
        "trailing_stop": s.get("trailing_stop"),
        "trailing_atr_mult": s.get("trailing_atr_mult"),
        "atr_window": s.get("atr_window"),
        "stop_priority": s.get("stop_priority"),
        "cost_bps": s.get("cost_bps"),
        "slippage_bps": s.get("slippage_bps"),
        "execution_delay": s.get("execution_delay"),
        "vol_target": s.get("vol_target"),
        "vol_lookback": s.get("vol_lookback"),
        "max_leverage": s.get("max_leverage"),
        "portfolio_weighting": s.get("portfolio_weighting"),
        "weight_lookback": s.get("weight_lookback"),
        "model": s.get("model"),
        "input": s.get("input"),
        # Wavelet kernel comparison metadata (may be absent for most runs)
        "wavelet_kernel": s.get("wavelet_kernel"),
        "wavelet_ar_p": s.get("wavelet_ar_p"),
    }

    base.update(_components_from_summary(s))

    # Prefer portfolio_metrics if available; else use single-asset metrics.
    if is_portfolio:
        base.update(_flatten_metrics("m_", portfolio_metrics))
    else:
        base.update(_flatten_metrics("m_", metrics))

    if trade_metrics:
        base.update(_flatten_metrics("t_", trade_metrics))

    return base


def _asset_rows_for_run(run_dir: Path, *, score_col: str) -> pd.DataFrame:
    """Build per-asset rows for a run.

    - For portfolio runs, use assets_metrics.csv if available.
    - For single-asset runs, use the run-level metrics and asset name.
    """
    s = _load_json(run_dir / "backtest_summary.json")
    is_portfolio = "portfolio_metrics" in s

    rows: list[dict] = []

    if is_portfolio:
        p = run_dir / "assets_metrics.csv"
        if p.exists():
            try:
                am = pd.read_csv(p)
                # Normalize column names into m_* prefixes where appropriate.
                for _, r in am.iterrows():
                    d = {"run": run_dir.name, "run_dir": str(run_dir), "asset": str(r.get("asset"))}
                    for k, v in r.to_dict().items():
                        if k == "asset":
                            continue
                        d[f"m_{k}"] = v
                    rows.append(d)
            except Exception:
                pass
    else:
        asset = s.get("asset")
        # Prefer recomputed run metrics if present in summary; else compute quickly from curve.
        m = s.get("metrics") if isinstance(s.get("metrics"), dict) else {}
        curve = _read_curve(run_dir / "equity_curve.csv")
        if not curve.empty:
            try:
                m = compute_metrics(curve)
            except Exception:
                pass
        d = {"run": run_dir.name, "run_dir": str(run_dir), "asset": str(asset) if asset is not None else run_dir.name}
        for k, v in (m or {}).items():
            d[f"m_{k}"] = v
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Ensure score col numeric for ranking.
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    return df


def _write_html_report(
    df: pd.DataFrame,
    out_html: Path,
    *,
    score_col: str,
    assets_df: pd.DataFrame | None = None,
    asset: str | None = None,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return

    dff = df.copy()

    # If an asset is specified, switch to a per-asset comparison view.
    # This compares the chosen asset's metrics across runs, while still showing
    # the combination of components used in each run.
    if asset is not None and assets_df is not None and not assets_df.empty:
        dff = _merge_asset_metrics(dff, assets_df, asset=str(asset))
        if not dff.empty:
            dff["feature_combo"] = dff.apply(lambda r: _combo_from_components(r.to_dict()), axis=1)

    # Clean up for plotting
    if score_col not in dff.columns:
        score_col = "m_total_return" if "m_total_return" in dff.columns else dff.columns[-1]

    # Sort best to worst
    dff[score_col] = pd.to_numeric(dff[score_col], errors="coerce")
    dff = dff.sort_values(score_col, ascending=False)

    # Components matrix
    comp_cols = [
        "long_only",
        "uses_proba_signal",
        "uses_stop_loss",
        "uses_take_profit",
        "uses_trailing_stop",
        "uses_trailing_atr",
        "uses_vol_target",
        "uses_wavelet",
    ]
    comp_cols = [c for c in comp_cols if c in dff.columns]

    comp = dff[comp_cols].astype(float) if comp_cols else pd.DataFrame(index=dff.index)

    # Asset-focused view: Sharpe vs component combo (single panel + components heatmap).
    if asset is not None:
        # If we failed to build an asset-specific table, still write a minimal message.
        if dff.empty:
            out_html.write_text(
                f"<html><body><h2>No rows found for asset={asset!r}</h2></body></html>",
                encoding="utf-8",
            )
            return

        # Exclude rows that would plot as zero-height (or invalid) bars.
        # This keeps the chart readable when many variants produced no trades.
        score_vals = pd.to_numeric(dff[score_col], errors="coerce")
        ok = np.isfinite(score_vals.to_numpy()) & (np.abs(score_vals.to_numpy()) > 1e-12)
        dff = dff.loc[ok].copy()
        if dff.empty:
            out_html.write_text(
                f"<html><body><h2>No non-zero {score_col} values for asset={asset!r}</h2></body></html>",
                encoding="utf-8",
            )
            return

        comp_cols = [
            "long_only",
            "uses_proba_signal",
            "uses_stop_loss",
            "uses_take_profit",
            "uses_trailing_stop",
            "uses_trailing_atr",
            "uses_vol_target",
            "uses_wavelet",
        ]
        comp_cols = [c for c in comp_cols if c in dff.columns]
        comp = dff[comp_cols].astype(float) if comp_cols else pd.DataFrame(index=dff.index)

        # Build a readable x label that starts with the combo (as requested) and includes run name.
        combo_label = (
            dff["feature_combo"].astype(str)
            + "<br><span style='font-size:10px'>"
            + dff["run"].astype(str)
            + "</span>"
        )

        # To keep x-axis labels readable, render the bar chart and the components
        # heatmap as *two separate figures* stacked in the HTML with extra spacing.
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=combo_label,
                y=dff[score_col],
                name=score_col,
                hovertemplate=(
                    "asset=%{customdata[0]}<br>run=%{customdata[1]}<br>combo=%{customdata[2]}<br>sharpe=%{y:.4f}<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        np.asarray([str(asset)] * len(dff)),
                        dff["run"].astype(str).to_numpy(),
                        dff["feature_combo"].astype(str).to_numpy(),
                    ],
                    axis=1,
                ),
            )
        )
        fig_bar.update_layout(
            template="plotly_white",
            height=650,
            title=f"{asset}: Sharpe ratio by strategy component combo",
            margin=dict(l=60, r=30, t=80, b=260),
        )
        fig_bar.update_yaxes(title_text="Sharpe ratio (annualized)")
        fig_bar.update_xaxes(
            title_text="Component combo (+ run)",
            tickangle=-35,
            automargin=True,
        )

        fig_comp = None
        if not comp.empty:
            fig_comp = go.Figure(
                data=[
                    go.Heatmap(
                        z=comp.to_numpy().T,
                        x=dff["run"],
                        y=comp.columns.tolist(),
                        colorscale=[[0.0, "#f0f0f0"], [1.0, "#1f77b4"]],
                        showscale=False,
                        name="components",
                    )
                ]
            )
            fig_comp.update_layout(
                template="plotly_white",
                height=420,
                title="Components used (1=yes)",
                margin=dict(l=60, r=30, t=80, b=160),
            )
            fig_comp.update_xaxes(title_text="Backtest run", tickangle=-25, automargin=True)
            fig_comp.update_yaxes(title_text="Component")

        # Compose a single HTML file with both figures.
        html_parts = [
            fig_bar.to_html(full_html=False, include_plotlyjs="cdn"),
        ]
        if fig_comp is not None:
            html_parts.append("<div style='height:40px'></div>")
            html_parts.append(fig_comp.to_html(full_html=False, include_plotlyjs=False))

        page = "".join(
            [
                "<html><head><meta charset='utf-8'></head><body>",
                *html_parts,
                "</body></html>",
            ]
        )
        out_html.write_text(page, encoding="utf-8")
        return

    # Optional 4th panel: best run per asset (multi-market view).
    has_assets = assets_df is not None and isinstance(assets_df, pd.DataFrame) and not assets_df.empty and "asset" in assets_df.columns
    n_rows = 4 if has_assets else 3

    titles = [
        f"Strategy score by run ({score_col})",
        "Components used (1=yes)",
        "Risk/return scatter",
    ]
    if has_assets:
        titles.append(f"Best score per asset ({score_col})")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        # Don't share x-axes: bar/heatmap use categorical run names, scatter uses numeric drawdown.
        shared_xaxes=False,
        vertical_spacing=0.06,
        row_heights=([0.40, 0.22, 0.26, 0.22] if has_assets else [0.45, 0.25, 0.30]),
        subplot_titles=tuple(titles),
    )

    # Bar: score
    fig.add_trace(
        go.Bar(x=dff["run"], y=dff[score_col], name=score_col),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Backtest run", row=1, col=1)
    fig.update_yaxes(title_text=_pretty_metric_name(str(score_col)), row=1, col=1)

    # Heatmap: components
    if not comp.empty:
        fig.add_trace(
            go.Heatmap(
                z=comp.to_numpy().T,
                x=dff["run"],
                y=comp.columns.tolist(),
                colorscale=[[0.0, "#f0f0f0"], [1.0, "#1f77b4"]],
                showscale=False,
                name="components",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Backtest run", row=2, col=1)
        fig.update_yaxes(title_text="Component", row=2, col=1)

    # Scatter: drawdown vs return (or score)
    x_dd = "m_max_drawdown" if "m_max_drawdown" in dff.columns else None
    y_ret = "m_total_return" if "m_total_return" in dff.columns else score_col

    if x_dd is not None:
        fig.add_trace(
            go.Scatter(
                x=dff[x_dd],
                y=dff[y_ret],
                mode="markers+text",
                text=dff["run"],
                textposition="top center",
                name="runs",
                marker=dict(size=10, opacity=0.85),
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(title_text="max_drawdown", row=3, col=1)
    fig.update_yaxes(title_text=_pretty_metric_name(str(y_ret)), row=3, col=1)

    # Asset-level best score chart
    if has_assets and assets_df is not None:
        adf = assets_df.copy()
        if score_col not in adf.columns:
            # Can't plot without the score metric.
            pass
        else:
            adf[score_col] = pd.to_numeric(adf[score_col], errors="coerce")
            adf = adf.dropna(subset=[score_col])
            if not adf.empty:
                # Pick best run per asset
                best = adf.sort_values(score_col, ascending=False).groupby("asset", as_index=False).head(1)
                best = best.sort_values(score_col, ascending=False)
                fig.add_trace(
                    go.Bar(
                        x=best["asset"],
                        y=best[score_col],
                        name="best_per_asset",
                        hovertext=best["run"],
                        hovertemplate="asset=%{x}<br>score=%{y}<br>best_run=%{hovertext}<extra></extra>",
                    ),
                    row=4,
                    col=1,
                )
                fig.update_xaxes(title_text="Asset / market", row=4, col=1)
                fig.update_yaxes(title_text=_pretty_metric_name(str(score_col)), row=4, col=1)

                # Footnote with best strategy per asset.
                lines: list[str] = []
                for _, r in best.iterrows():
                    asset = str(r.get("asset"))
                    run = str(r.get("run"))
                    val = r.get(score_col)
                    if val is None:
                        val_s = "nan"
                    else:
                        try:
                            val_f = float(val)
                            val_s = f"{val_f:.3f}"
                        except Exception:
                            val_s = str(val)
                    lines.append(f"{asset}: {run} ({score_col}={val_s})")

                footnote = "<b>Best strategy per asset</b><br>" + "<br>".join(lines)
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.18,
                    text=footnote,
                    showarrow=False,
                    align="left",
                    font=dict(size=12, color="#333"),
                )

    # Extra bottom margin so the footnote (when present) is visible.
    fig.update_layout(
        template="plotly_white",
        height=(1200 if has_assets else 950),
        title="Backtest strategy comparison",
        margin=dict(l=60, r=30, t=80, b=(240 if has_assets else 80)),
    )

    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a comparison report across backtest output folders")
    p.add_argument("--in-root", default=".", help="Root folder to scan (default: .)")
    p.add_argument("--glob", default="bt_*", help="Glob pattern for backtest folders (default: bt_*)")
    p.add_argument("--out-dir", default="backtest_report", help="Output directory")
    p.add_argument(
        "--score",
        default="m_sharpe",
        help="Column to rank/plot (e.g., m_total_return | m_sharpe | m_cagr)",
    )
    p.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Recompute metrics from equity curve CSVs (recommended if metrics definitions changed)",
    )
    p.add_argument(
        "--asset",
        default=None,
        help="If set, build an asset-specific comparison report using per-asset metrics (e.g., EURUSD=X)",
    )
    p.add_argument(
        "--exclude-crypto",
        action="store_true",
        help="Exclude runs that include crypto assets (e.g., BTC-USD, ETH-USD)",
    )

    args = p.parse_args(argv)

    root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(root, str(args.glob))
    if args.exclude_crypto:
        runs = [r for r in runs if not _should_exclude_run_for_crypto(r)]
    if not runs:
        raise SystemExit(f"No backtest runs found in {root} matching {args.glob} with backtest_summary.json")

    rows = [_build_report_rows(r, recompute_metrics=bool(args.recompute_metrics)) for r in runs]
    df = pd.DataFrame(rows)

    # Asset-level table (for multi-market view)
    assets_rows: list[pd.DataFrame] = []
    for r in runs:
        assets_rows.append(_asset_rows_for_run(r, score_col=str(args.score)))
    assets_df = pd.concat([x for x in assets_rows if x is not None and not x.empty], axis=0, ignore_index=True) if assets_rows else pd.DataFrame()

    # Optionally exclude crypto assets from the per-asset view.
    if args.exclude_crypto and not assets_df.empty and "asset" in assets_df.columns:
        assets_df = assets_df[~assets_df["asset"].astype(str).apply(_is_crypto_symbol)].copy()

    # Write CSV for easy inspection
    out_csv = out_dir / "strategy_comparison.csv"
    df.to_csv(out_csv, index=False)

    # If an asset is requested, also write a filtered CSV with just that asset's metrics
    # (useful because the full run-level CSV includes non-matching assets).
    asset = str(args.asset) if args.asset else None
    if asset:
        dff = _merge_asset_metrics(df, assets_df, asset=asset)
        if not dff.empty:
            dff = dff.copy()
            dff["feature_combo"] = dff.apply(lambda r: _combo_from_components(r.to_dict()), axis=1)
            out_asset_csv = out_dir / f"strategy_comparison__asset_{re.sub(r'[^A-Za-z0-9_=\-]', '_', asset)}.csv"
            dff.to_csv(out_asset_csv, index=False)

    # Write HTML chart
    out_html = out_dir / "strategy_report.html"
    _write_html_report(df, out_html, score_col=str(args.score), assets_df=assets_df, asset=asset)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
