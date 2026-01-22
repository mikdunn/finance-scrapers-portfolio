"""Strategy simulator / backtester.

Loads a trained sklearn Pipeline (from projects/ml_train.py outputs), produces
model signals on historical OHLCV features, and simulates an equity curve with
simple transaction costs.

Supports:
- Single-asset backtest from one CSV
- Multi-asset portfolio backtest from a directory of CSVs

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.backtest import (
    BacktestConfig,
    compute_equity_curve,
    compute_metrics,
    compute_trade_metrics,
    trades_from_positions,
    simulate_ohlc,
)
from utils.ml_features import build_features


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    date_col = None
    for c in ("Date", "Datetime", "date", "datetime"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        if df.columns.size and str(df.columns[0]).lower().startswith("unnamed"):
            date_col = df.columns[0]

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    return df


def _asset_name_from_path(p: Path) -> str:
    stem = p.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _gather_inputs(in_csv: str | None, in_dir: str | None) -> list[Path]:
    if in_csv:
        p = Path(in_csv)
        if not p.exists():
            raise FileNotFoundError(str(p))
        return [p]

    if not in_dir:
        raise ValueError("Provide --in-csv or --in-dir")

    d = Path(in_dir)
    if not d.exists():
        raise FileNotFoundError(str(d))

    exclude_names = {
        "summary.json",
        "macro_features.csv",
        "extra_features_raw.csv",
        "portfolio_weights.csv",
        "spectral_clusters.csv",
        "dmd_eigs.csv",
    }

    out: list[Path] = []
    for p in d.glob("*.csv"):
        if p.name.lower() in exclude_names:
            continue
        out.append(p)

    # One sharding level (matches ml_train behavior)
    for child in d.iterdir():
        if child.is_dir():
            for p in child.glob("*.csv"):
                if p.name.lower() in exclude_names:
                    continue
                out.append(p)

    return sorted(set(out))


def _load_model(model_path: Path):
    import joblib

    return joblib.load(model_path)


def _expected_feature_names(pipe) -> list[str] | None:
    # sklearn sets feature_names_in_ after fit; Pipeline may expose it.
    for obj in (pipe, getattr(pipe, "named_steps", {}).get("imputer"), getattr(pipe, "named_steps", {}).get("model")):
        if obj is None:
            continue
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


def _predict_proba_safe(pipe, X: pd.DataFrame) -> np.ndarray | None:
    try:
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)
    except Exception:
        return None
    return None


def _proba_to_signal(pipe, proba: np.ndarray, *, mode: str, enter: float) -> pd.Series:
    """Map predict_proba output to {-1,0,1} signals using the estimator's class labels."""
    if proba is None or not isinstance(proba, np.ndarray) or proba.ndim != 2:
        raise ValueError("predict_proba returned an unexpected shape")

    model = getattr(pipe, "named_steps", {}).get("model")
    classes = getattr(model, "classes_", None)
    if classes is None:
        # Fall back to 3-class convention
        classes = np.asarray([-1, 0, 1])
    classes = np.asarray(classes)

    # Pull probs for buy/sell if present
    p_buy = None
    p_sell = None
    for i, c in enumerate(classes.tolist()):
        if int(c) == 1:
            p_buy = proba[:, i]
        elif int(c) == -1:
            p_sell = proba[:, i]

    if p_buy is None and classes.size == 3 and set(classes.tolist()) == {0, 1, 2}:
        # XGB-style 0,1,2 mapped to -1,0,1
        p_sell = proba[:, 0]
        p_buy = proba[:, 2]

    if p_buy is None and p_sell is None:
        raise ValueError("Could not locate buy/sell classes in predict_proba output")

    enter = float(enter)
    sig = np.zeros(proba.shape[0], dtype=float)
    if p_buy is not None:
        sig[p_buy >= enter] = 1.0
    if (mode or "long_short").strip().lower() not in {"long", "long_only", "long-only"}:
        if p_sell is not None:
            sig[p_sell >= enter] = -1.0
    return pd.Series(sig)


def _align_to_model_features(X: pd.DataFrame, pipe) -> pd.DataFrame:
    names = _expected_feature_names(pipe)
    if not names:
        return X

    out = X.copy()
    for c in names:
        if c not in out.columns:
            out[c] = 0.0

    out = out.loc[:, names]
    return out


def _infer_task(pipe, task_arg: str | None) -> str:
    if task_arg and task_arg.strip().lower() not in {"auto", ""}:
        t = task_arg.strip().lower()
        if t in {"classification", "cls", "class"}:
            return "classification"
        if t in {"regression", "reg", "return"}:
            return "regression"

    model = getattr(pipe, "named_steps", {}).get("model")
    est_type = getattr(model, "_estimator_type", None)
    if est_type == "classifier":
        return "classification"
    if est_type == "regressor":
        return "regression"

    # Fallback heuristic
    if hasattr(model, "predict_proba"):
        return "classification"
    return "regression"


def _map_predictions_to_signal(pred: np.ndarray, *, task: str, threshold: float) -> pd.Series:
    p = pd.Series(pred)

    if task == "regression":
        thr = float(threshold)
        sig = pd.Series(index=p.index, dtype=float)
        sig[p > thr] = 1.0
        sig[p < -thr] = -1.0
        sig[(p <= thr) & (p >= -thr)] = 0.0
        return sig

    # classification
    # If model emits 0..2 (xgboost multi-class), map to -1,0,1.
    uniq = set(pd.to_numeric(p, errors="coerce").dropna().astype(int).unique().tolist())
    if uniq and uniq.issubset({0, 1, 2}) and not uniq.issubset({-1, 0, 1}):
        p = p.replace({0: -1, 1: 0, 2: 1})
    return pd.to_numeric(p, errors="coerce").fillna(0.0).clip(-1, 1).round(0)


def _predict_signals_for_asset(
    pipe,
    df_raw: pd.DataFrame,
    *,
    asset: str | None,
    task: str,
    threshold: float,
    signal_source: str,
    proba_enter: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if df_raw is None or df_raw.empty:
        raise ValueError("Empty dataset")

    if "Close" not in df_raw.columns:
        raise ValueError("Input data must have a Close column")

    X = build_features(df_raw)

    # If the model was trained in multi-asset mode, it likely expects asset_* one-hot columns.
    # We create them here (then align to exact expected model columns).
    if asset is not None:
        X2 = X.copy()
        X2["asset"] = asset
        X2 = pd.get_dummies(X2, columns=["asset"], prefix="asset", dtype=float)
        X = X2

    X = X.dropna(how="all")

    X_aligned = _align_to_model_features(X, pipe)

    signal_source = (signal_source or "predict").strip().lower()
    if task == "classification" and signal_source in {"proba", "prob", "predict_proba"}:
        proba = _predict_proba_safe(pipe, X_aligned)
        if proba is None:
            raise ValueError("Model/pipeline does not support predict_proba")
        sig = _proba_to_signal(pipe, proba, mode="long_short", enter=float(proba_enter))
        sig.index = X_aligned.index
    else:
        pred = pipe.predict(X_aligned)
        sig = _map_predictions_to_signal(np.asarray(pred), task=task, threshold=float(threshold))
        sig.index = X_aligned.index

    close = pd.to_numeric(df_raw["Close"], errors="coerce").reindex(sig.index)
    close = close.dropna()
    sig = sig.reindex(close.index).fillna(0.0)

    return close, sig, X_aligned


def _write_equity_plot_html(path: Path, curve: pd.DataFrame, title: str) -> None:
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.index, y=curve["equity"], mode="lines", name="Equity"))
        fig.update_layout(title=title, template="plotly_white", height=550)
        fig.write_html(str(path))
    except Exception:
        # Plot is optional.
        return


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backtest a trained ML model with simple trading simulation")

    p.add_argument("--model", required=True, help="Path to model.joblib (from ML outputs)")
    p.add_argument("--in-csv", default=None, help="Input OHLCV+feature CSV (from market analyzer or hub)")
    p.add_argument("--in-dir", default=None, help="Input directory of per-symbol CSVs")
    p.add_argument("--out-dir", default="backtest_outputs", help="Where to write backtest artifacts")

    p.add_argument("--task", default="auto", help="auto | classification | regression")
    p.add_argument("--mode", default="long_short", help="long_short | long_only")
    p.add_argument("--threshold", type=float, default=0.003, help="Regression threshold for mapping predictions to signals")
    p.add_argument("--signal-source", default="predict", help="predict | proba (classification only)")
    p.add_argument("--proba-enter", type=float, default=0.55, help="Entry probability threshold when using --signal-source proba")

    p.add_argument("--price", default="close", help="Price used for returns: close | open")

    # Next realism step: explicit trade accounting with stops/TPs (uses OHLC)
    p.add_argument("--stop-loss", type=float, default=None, help="Stop loss as a fraction (e.g., 0.03 = 3%).")
    p.add_argument("--take-profit", type=float, default=None, help="Take profit as a fraction (e.g., 0.06 = 6%).")
    p.add_argument("--trailing-stop", type=float, default=None, help="Trailing stop as a fraction (e.g., 0.03 = 3%).")
    p.add_argument("--trailing-atr-mult", type=float, default=None, help="ATR trailing stop multiple (e.g., 3.0).")
    p.add_argument("--atr-window", type=int, default=14, help="ATR window (bars) for --trailing-atr-mult.")
    p.add_argument(
        "--stop-priority",
        default="conservative",
        help="When stop-loss and take-profit both hit in same bar: conservative|stop|take_profit",
    )

    p.add_argument("--vol-target", type=float, default=None, help="Annualized vol target for position sizing (e.g., 0.20). Off if omitted.")
    p.add_argument("--vol-lookback", type=int, default=20, help="Lookback window for realized vol in vol targeting")
    p.add_argument("--max-leverage", type=float, default=3.0, help="Max leverage when vol targeting")

    p.add_argument("--cost-bps", type=float, default=5.0, help="Round-trip transaction cost per 1x position change (bps)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Extra slippage per 1x position change (bps)")
    p.add_argument("--delay", type=int, default=1, help="Execution delay in bars (default 1 bar)")

    p.add_argument("--max-assets", type=int, default=None, help="Limit number of assets when using --in-dir")

    p.add_argument("--portfolio-weighting", default="equal", help="Multi-asset portfolio weighting: equal | inverse_vol")
    p.add_argument("--weight-lookback", type=int, default=60, help="Lookback (bars) for inverse-vol portfolio weights")

    args = p.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_model(model_path)
    task = _infer_task(pipe, args.task)

    inputs = _gather_inputs(args.in_csv, args.in_dir)
    if args.max_assets is not None:
        inputs = inputs[: int(args.max_assets)]

    if not inputs:
        raise SystemExit("No input CSVs found")

    cfg = BacktestConfig(
        mode=str(args.mode),
        execution_delay=int(args.delay),
        cost_bps=float(args.cost_bps),
        slippage_bps=float(args.slippage_bps),
        initial_equity=1.0,
        vol_target=(float(args.vol_target) if args.vol_target is not None else None),
        vol_lookback=int(args.vol_lookback),
        max_leverage=float(args.max_leverage),
    )

    # Decide if model expects asset dummies: if any expected feature starts with asset_
    expected = _expected_feature_names(pipe) or []
    expects_asset_dummies = any(str(c).startswith("asset_") for c in expected)

    price_choice = (args.price or "close").strip().lower()

    def _select_price_frame(df: pd.DataFrame) -> pd.DataFrame:
        if price_choice == "open" and "Open" in df.columns:
            # Use Open as the tradable mark price
            df2 = df.copy()
            df2["Close"] = pd.to_numeric(df2["Open"], errors="coerce")
            return df2
        return df

    def _use_trade_sim() -> bool:
        return (
            (args.stop_loss is not None)
            or (args.take_profit is not None)
            or (args.trailing_stop is not None)
            or (args.trailing_atr_mult is not None)
        )

    # Single-asset
    if len(inputs) == 1 and args.in_csv:
        path = inputs[0]
        df = _select_price_frame(_load_csv(path))
        asset = _asset_name_from_path(path) if expects_asset_dummies else None

        close, sig, _X = _predict_signals_for_asset(
            pipe,
            df,
            asset=asset,
            task=task,
            threshold=float(args.threshold),
            signal_source=str(args.signal_source),
            proba_enter=float(args.proba_enter),
        )

        if _use_trade_sim():
            # For stop/TP realism, we simulate using OHLC bars; enforce open marking.
            curve, trades = simulate_ohlc(
                df,
                sig,
                cfg=cfg,
                mark="open",
                stop_loss=args.stop_loss,
                take_profit=args.take_profit,
                trailing_stop=args.trailing_stop,
                trailing_atr_mult=args.trailing_atr_mult,
                atr_window=int(args.atr_window),
                stop_priority=str(args.stop_priority),
            )
            metrics = compute_metrics(curve)
            curve.to_csv(out_dir / "equity_curve.csv")
            trades.to_csv(out_dir / "trades.csv", index=False)
            _write_equity_plot_html(out_dir / "equity_curve.html", curve, title=f"Equity Curve: {asset or path.stem}")
        else:
            curve = compute_equity_curve(close, sig, cfg=cfg)
            metrics = compute_metrics(curve)
            trades = trades_from_positions(curve["close"], curve["position"])
            curve.to_csv(out_dir / "equity_curve.csv")
            trades.to_csv(out_dir / "trades.csv", index=False)
            _write_equity_plot_html(out_dir / "equity_curve.html", curve, title=f"Equity Curve: {asset or path.stem}")

        trade_metrics = compute_trade_metrics(trades)

        summary = {
            "mode": cfg.mode,
            "task": task,
            "signal_source": str(args.signal_source),
            "price": price_choice,
            "stop_loss": args.stop_loss,
            "take_profit": args.take_profit,
            "trailing_stop": args.trailing_stop,
            "trailing_atr_mult": args.trailing_atr_mult,
            "atr_window": int(args.atr_window),
            "stop_priority": str(args.stop_priority),
            "cost_bps": cfg.cost_bps,
            "slippage_bps": cfg.slippage_bps,
            "execution_delay": cfg.execution_delay,
            "vol_target": cfg.vol_target,
            "vol_lookback": cfg.vol_lookback,
            "max_leverage": cfg.max_leverage,
            "model": str(model_path),
            "input": str(path),
            "asset": asset or _asset_name_from_path(path),
            "metrics": metrics,
            "trade_metrics": trade_metrics,
        }
        with open(out_dir / "backtest_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Wrote: {out_dir / 'equity_curve.csv'}")
        print(f"Wrote: {out_dir / 'backtest_summary.json'}")
        return 0

    # Multi-asset portfolio
    per_asset_metrics: list[dict] = []
    per_asset_returns: list[pd.Series] = []

    for path in inputs:
        try:
            df = _select_price_frame(_load_csv(path))
            asset_name = _asset_name_from_path(path)
            asset = asset_name if expects_asset_dummies else None
            close, sig, _X = _predict_signals_for_asset(
                pipe,
                df,
                asset=asset,
                task=task,
                threshold=float(args.threshold),
                signal_source=str(args.signal_source),
                proba_enter=float(args.proba_enter),
            )

            if _use_trade_sim():
                curve, trades = simulate_ohlc(
                    df,
                    sig,
                    cfg=cfg,
                    mark="open",
                    stop_loss=args.stop_loss,
                    take_profit=args.take_profit,
                    trailing_stop=args.trailing_stop,
                    trailing_atr_mult=args.trailing_atr_mult,
                    atr_window=int(args.atr_window),
                    stop_priority=str(args.stop_priority),
                )
                m = compute_metrics(curve)
            else:
                curve = compute_equity_curve(close, sig, cfg=cfg)
                m = compute_metrics(curve)
                trades = trades_from_positions(curve["close"], curve["position"])
            m["asset"] = asset_name
            tm = compute_trade_metrics(trades)
            # Flatten a few headline trade metrics into the per-asset metrics table.
            m["n_trades"] = tm.get("n_trades")
            m["win_rate"] = tm.get("win_rate")
            m["profit_factor"] = tm.get("profit_factor")
            per_asset_metrics.append(m)

            # Store strategy returns for portfolio aggregation
            sret = curve["strategy_ret"].rename(asset_name)
            per_asset_returns.append(sret)

            # Write per-asset artifacts
            asset_dir = out_dir / "assets" / asset_name
            asset_dir.mkdir(parents=True, exist_ok=True)
            curve.to_csv(asset_dir / "equity_curve.csv")
            trades.to_csv(asset_dir / "trades.csv", index=False)
            _write_equity_plot_html(asset_dir / "equity_curve.html", curve, title=f"Equity Curve: {asset_name}")
        except Exception as e:
            per_asset_metrics.append({"asset": _asset_name_from_path(path), "error": str(e)})

    if not per_asset_returns:
        raise SystemExit("No assets produced returns; cannot build portfolio")

    panel = pd.concat(per_asset_returns, axis=1).sort_index()

    weighting = (args.portfolio_weighting or "equal").strip().lower()
    if weighting in {"inv_vol", "inverse_vol", "inverse-vol", "risk_parity"}:
        lb = max(5, int(args.weight_lookback))
        vol = panel.rolling(lb, min_periods=lb).std(ddof=0)
        w = 1.0 / vol.replace(0.0, np.nan)
        w = w.div(w.sum(axis=1), axis=0)
        # Fallback to equal weights where vol not available
        w = w.where(w.notna(), other=0.0)
        row_sum = w.sum(axis=1)
        w = w.div(row_sum.replace(0.0, np.nan), axis=0)
        w = w.fillna(1.0 / panel.shape[1])
    else:
        w = pd.DataFrame(1.0 / panel.shape[1], index=panel.index, columns=panel.columns)

    port_ret = (panel * w).sum(axis=1, skipna=True).rename("strategy_ret")

    # Build a portfolio equity curve
    port_curve = pd.DataFrame({"strategy_ret": port_ret}).dropna()
    equity = [1.0]
    for r in port_curve["strategy_ret"].iloc[1:]:
        equity.append(equity[-1] * (1.0 + float(r)))
    port_curve["equity"] = pd.Series(equity, index=port_curve.index)

    # Add synthetic close (not meaningful) for compatibility with metrics; we’ll derive from equity.
    port_curve["close"] = port_curve["equity"]

    portfolio_metrics = compute_metrics(port_curve)

    port_curve.to_csv(out_dir / "portfolio_equity_curve.csv")
    _write_equity_plot_html(out_dir / "portfolio_equity_curve.html", port_curve, title="Portfolio Equity Curve")

    w.to_csv(out_dir / "portfolio_weights.csv")

    pd.DataFrame(per_asset_metrics).to_csv(out_dir / "assets_metrics.csv", index=False)

    summary = {
        "mode": cfg.mode,
        "task": task,
        "signal_source": str(args.signal_source),
        "price": price_choice,
        "stop_loss": args.stop_loss,
        "take_profit": args.take_profit,
        "trailing_stop": args.trailing_stop,
        "trailing_atr_mult": args.trailing_atr_mult,
        "atr_window": int(args.atr_window),
        "stop_priority": str(args.stop_priority),
        "cost_bps": cfg.cost_bps,
        "slippage_bps": cfg.slippage_bps,
        "execution_delay": cfg.execution_delay,
        "vol_target": cfg.vol_target,
        "vol_lookback": cfg.vol_lookback,
        "max_leverage": cfg.max_leverage,
        "model": str(model_path),
        "inputs": [str(p) for p in inputs],
        "expects_asset_dummies": bool(expects_asset_dummies),
        "portfolio_weighting": weighting,
        "weight_lookback": int(args.weight_lookback),
        "portfolio_metrics": portfolio_metrics,
    }
    with open(out_dir / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_dir / 'portfolio_equity_curve.csv'}")
    print(f"Wrote: {out_dir / 'assets_metrics.csv'}")
    print(f"Wrote: {out_dir / 'backtest_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
