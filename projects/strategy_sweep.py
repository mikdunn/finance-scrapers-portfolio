"""Strategy sweep / experiments.

Goal: generate *many* backtest runs for a single asset by combining:
- different ML "core" models (tree boosting, random forest, neural-net MLP)
- optional signal filters based on repo utilities (FFT dominant cycle, DMD)
- optional regime gating via HMM (if hmmlearn is installed)
- risk-mitigation knobs already supported by the backtester (vol targeting, stops)

It writes each variant into a bt_* folder so it can be compared by
`projects/backtest_report.py`.

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.backtest import (
    BacktestConfig,
    compute_equity_curve,
    compute_metrics,
    compute_trade_metrics,
    simulate_ohlc,
    trades_from_positions,
)
from utils.dmd import exact_dmd, eigenvalue_to_frequency
from utils.fourier import rolling_dominant_period
from utils.wavelets import cwt_band_gate_from_returns, daubechies_dwt_band_gate_from_returns
from utils.ml_features import LabelSpec, build_features, make_labels


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    date_col = None
    for c in ("Date", "Datetime", "date", "datetime"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None and df.columns.size and str(df.columns[0]).lower().startswith("unnamed"):
        date_col = df.columns[0]

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    return df


def _write_equity_plot_html(path: Path, curve: pd.DataFrame, title: str) -> None:
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.index, y=curve["equity"], mode="lines", name="Equity"))
        fig.update_layout(title=title, template="plotly_white", height=550)
        fig.write_html(str(path))
    except Exception:
        return


def _proba_to_signal(classes: np.ndarray, proba: np.ndarray, *, enter: float, allow_short: bool) -> np.ndarray:
    """Map predict_proba to {-1,0,1} using class labels."""
    classes = np.asarray(classes)
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError("predict_proba must be 2D")

    p_buy = None
    p_sell = None
    for i, c in enumerate(classes.tolist()):
        try:
            ci = int(c)
        except Exception:
            continue
        if ci == 1:
            p_buy = proba[:, i]
        elif ci == -1:
            p_sell = proba[:, i]

    if p_buy is None and p_sell is None:
        raise ValueError("Could not locate buy/sell classes")

    sig = np.zeros(proba.shape[0], dtype=float)
    thr = float(enter)
    if p_buy is not None:
        sig[p_buy >= thr] = 1.0
    if allow_short and p_sell is not None:
        sig[p_sell >= thr] = -1.0
    return sig


def _hmm_states(returns: pd.Series, *, train_mask: np.ndarray, n_states: int = 3) -> pd.Series | None:
    """Fit a simple Gaussian HMM on returns and return state sequence.

    Returns None if hmmlearn is not available.
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except Exception:
        return None

    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    X = r.to_numpy().reshape(-1, 1)

    # Fit on train only.
    X_train = X[train_mask]
    if X_train.shape[0] < 50:
        return None

    hmm = GaussianHMM(n_components=int(n_states), covariance_type="diag", n_iter=200, random_state=42)
    hmm.fit(X_train)
    states = hmm.predict(X)
    return pd.Series(states, index=r.index, name="hmm_state")


def _hmm_risk_off_mask(returns: pd.Series, states: pd.Series, *, train_mask: np.ndarray) -> pd.Series:
    """Choose a risk-off state and return mask True when risk-off."""
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    st = pd.to_numeric(states, errors="coerce")

    df = pd.DataFrame({"r": r, "s": st}).dropna()
    if df.empty:
        return pd.Series(False, index=r.index)

    # Compute per-state stats on train.
    df_train = df.loc[df.index[train_mask]] if train_mask.shape[0] == len(r.index) else df
    grp = df_train.groupby("s")
    stats = grp["r"].agg(["mean", "std", "count"]).reset_index()
    stats = stats[stats["count"] >= 10]
    if stats.empty:
        return pd.Series(False, index=r.index)

    # Risk-off: highest volatility (std). Break ties by lowest mean.
    stats = stats.sort_values(["std", "mean"], ascending=[False, True])
    risk_off_state = int(stats.iloc[0]["s"])

    out = (st == risk_off_state).reindex(r.index).fillna(False)
    return out.astype(bool)


def _fft_trade_mask(close: pd.Series, *, window: int = 128, min_p: float = 10.0, max_p: float = 80.0) -> pd.Series:
    """Simple FFT-based gate: only trade when a dominant period exists and is in a reasonable band."""
    per = rolling_dominant_period(close, window=window, min_period=min_p, max_period=max_p)
    ok = per.notna()
    return ok.reindex(close.index).fillna(False).astype(bool)


def _fit_ar_coeffs(returns: pd.Series, *, train_mask: np.ndarray, p: int = 5, ridge: float = 1e-6) -> np.ndarray | None:
    """Fit AR(p) coefficients on train returns via ridge-regularized least squares."""
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=float)
    mask = np.asarray(train_mask, dtype=bool)
    p = int(p)
    if p < 1:
        return None
    if r.size < (p + 20):
        return None
    if mask.size != r.size:
        mask = np.ones_like(r, dtype=bool)

    # Build full lag matrix once and then mask rows.
    # For t>=p: y_t = r_t, X_t = [r_{t-1}, ..., r_{t-p}]
    y = r[p:]
    X = np.column_stack([r[p - k - 1 : -k - 1] for k in range(p)])
    row_mask = mask[p:]
    X = X[row_mask]
    y = y[row_mask]
    if y.size < (p + 10):
        return None

    # Ridge: (X'X + λI)β = X'y
    XtX = X.T @ X
    XtX = XtX + float(ridge) * np.eye(p)
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except Exception:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(beta, dtype=float)


def _ar_residuals(returns: pd.Series, *, beta: np.ndarray | None) -> pd.Series:
    """Compute AR residuals r_t - beta·[r_{t-1}..r_{t-p}] for the full series."""
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    if beta is None or beta.size == 0:
        return r
    b = np.asarray(beta, dtype=float)
    p = int(b.size)
    arr = r.to_numpy(dtype=float)
    res = np.zeros_like(arr)
    # Start at t=p
    for t in range(p, arr.size):
        # lag vector: [r_{t-1}, ..., r_{t-p}]
        x = np.array([arr[t - k] for k in range(1, p + 1)], dtype=float)
        res[t] = arr[t] - float(np.dot(b, x))
    # Warmup stays 0
    return pd.Series(res, index=r.index, name=f"ar_resid_p{p}")


def _wavelet_gate_for_kernel(
    returns: pd.Series,
    *,
    kernel: str,
    window: int = 128,
    min_p: float = 10.0,
    max_p: float = 80.0,
    thr: float = 2.5,
) -> pd.Series:
    """Compute a wavelet-style trade gate for a given kernel name.

    Kernels:
    - dwt_db2 / dwt_db4 / dwt_db6 / dwt_db8
    - cwt_morlet / cwt_ricker
    """
    k = str(kernel or "").strip().lower()
    if k.startswith("dwt_db"):
        try:
            order = int(k.replace("dwt_db", "").strip())
        except Exception:
            order = 8
        return daubechies_dwt_band_gate_from_returns(
            returns,
            window=int(window),
            min_period=float(min_p),
            max_period=float(max_p),
            orders=(int(order),),
            min_energy_ratio=float(thr),
        ).rename("wavelet_ok")

    if k in {"cwt_morlet", "cwt_morlet2", "morlet", "morlet2"}:
        # CWT power ratios tend to be closer to ~1; use a slightly looser threshold.
        thr_cwt = float(min(thr, 1.8))
        return cwt_band_gate_from_returns(
            returns,
            window=int(window),
            min_period=float(min_p),
            max_period=float(max_p),
            kernel="morlet2",
            n_periods=48,
            min_power_ratio=float(thr_cwt),
        ).rename("wavelet_ok")

    if k in {"cwt_ricker", "ricker", "mexh"}:
        thr_cwt = float(min(thr, 1.8))
        return cwt_band_gate_from_returns(
            returns,
            window=int(window),
            min_period=float(min_p),
            max_period=float(max_p),
            kernel="ricker",
            n_periods=48,
            min_power_ratio=float(thr_cwt),
        ).rename("wavelet_ok")

    # Unknown kernel -> never trade.
    return pd.Series(False, index=returns.index, name="wavelet_ok")


def _dmd_trade_mask(returns: pd.Series, *, window: int = 60, lookback: int = 10, rank: int = 3) -> pd.Series:
    """Simple DMD-based gate: trade only when estimated growth is non-negative.

    We form a Hankel-like matrix from lagged returns over a trailing window.
    """
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    n = len(r)
    out = np.full(n, False, dtype=bool)

    w = int(window)
    lb = int(lookback)
    if n < max(w, lb + 5):
        return pd.Series(out, index=r.index)

    for i in range(w - 1, n):
        seg = r.iloc[i - w + 1 : i + 1].to_numpy(dtype=float)
        # Build lagged matrix: shape (lb, w-lb+1)
        t = w - lb + 1
        if t < 3:
            continue
        X = np.zeros((lb, t), dtype=float)
        for k in range(lb):
            X[k, :] = seg[k : k + t]

        try:
            res = exact_dmd(X, rank=rank)
            growth, _freq = eigenvalue_to_frequency(res.eigenvalues, dt=1.0)
            g = float(np.nanmax(np.real(growth))) if growth.size else 0.0
            out[i] = bool(np.isfinite(g) and g >= 0.0)
        except Exception:
            out[i] = False

    return pd.Series(out, index=r.index, name="dmd_trade_ok")


@dataclass(frozen=True)
class Variant:
    name: str
    core: str  # hgb|rf|mlp
    use_proba: bool
    proba_enter: float
    hmm_gate: bool
    fft_gate: bool
    wavelet_gate: bool
    dmd_gate: bool = False
    vol_target: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_atr_mult: float | None = None

    # Extra wavelet configuration (for kernel comparisons)
    wavelet_kernel: str | None = None
    wavelet_ar_p: int | None = None


def _make_variants() -> list[Variant]:
    cores = ["hgb", "rf", "mlp"]

    base: list[Variant] = []
    for core in cores:
        # Baseline
        base.append(
            Variant(
                name=f"{core}_baseline",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=False,
                fft_gate=False,
                wavelet_gate=False,
                dmd_gate=False,
                vol_target=None,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )

        # Risk overlays
        base.append(
            Variant(
                name=f"{core}_voltarget",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=False,
                fft_gate=False,
                wavelet_gate=False,
                dmd_gate=False,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )

        base.append(
            Variant(
                name=f"{core}_stops",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=False,
                fft_gate=False,
                wavelet_gate=False,
                dmd_gate=False,
                vol_target=None,
                stop_loss=0.01,
                take_profit=0.02,
                trailing_atr_mult=None,
            )
        )

        # "Spectral" (FFT) / DMD / HMM gates and combos
        base.append(
            Variant(
                name=f"{core}_hmm",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=True,
                fft_gate=False,
                wavelet_gate=False,
                dmd_gate=False,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )
        base.append(
            Variant(
                name=f"{core}_fft",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=False,
                fft_gate=True,
                wavelet_gate=False,
                dmd_gate=False,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )
        base.append(
            Variant(
                name=f"{core}_wavelet",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=False,
                fft_gate=False,
                wavelet_gate=True,
                wavelet_kernel="dwt_db8",
                wavelet_ar_p=None,
                dmd_gate=False,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )

        # Wavelet kernel comparison variants
        for k in ("dwt_db2", "dwt_db4", "dwt_db6", "dwt_db8", "cwt_morlet", "cwt_ricker"):
            base.append(
                Variant(
                    name=f"{core}_wv_{k}",
                    core=core,
                    use_proba=False,
                    proba_enter=0.55,
                    hmm_gate=False,
                    fft_gate=False,
                    wavelet_gate=True,
                    wavelet_kernel=k,
                    wavelet_ar_p=None,
                    dmd_gate=False,
                    vol_target=0.10,
                    stop_loss=None,
                    take_profit=None,
                    trailing_atr_mult=None,
                )
            )

        # AR prewhitened wavelet kernels (quick ARMA/ARIMA-ish experiment)
        for k in ("dwt_db8", "cwt_morlet", "cwt_ricker"):
            base.append(
                Variant(
                    name=f"{core}_wv_{k}_ar5",
                    core=core,
                    use_proba=False,
                    proba_enter=0.55,
                    hmm_gate=False,
                    fft_gate=False,
                    wavelet_gate=True,
                    wavelet_kernel=k,
                    wavelet_ar_p=5,
                    dmd_gate=False,
                    vol_target=0.10,
                    stop_loss=None,
                    take_profit=None,
                    trailing_atr_mult=None,
                )
            )
        base.append(
            Variant(
                name=f"{core}_dmd",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=False,
                fft_gate=False,
                wavelet_gate=False,
                dmd_gate=True,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )
        base.append(
            Variant(
                name=f"{core}_hmm_fft_dmd",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=True,
                fft_gate=True,
                wavelet_gate=False,
                dmd_gate=True,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )
        base.append(
            Variant(
                name=f"{core}_hmm_wavelet_dmd",
                core=core,
                use_proba=False,
                proba_enter=0.55,
                hmm_gate=True,
                fft_gate=False,
                wavelet_gate=True,
                dmd_gate=True,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )

        # Probability-based entry (classification proba gating)
        base.append(
            Variant(
                name=f"{core}_proba_hmm_fft",
                core=core,
                use_proba=True,
                proba_enter=0.60,
                hmm_gate=True,
                fft_gate=True,
                wavelet_gate=False,
                dmd_gate=False,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )

        base.append(
            Variant(
                name=f"{core}_proba_hmm_wavelet",
                core=core,
                use_proba=True,
                proba_enter=0.60,
                hmm_gate=True,
                fft_gate=False,
                wavelet_gate=True,
                dmd_gate=False,
                vol_target=0.10,
                stop_loss=None,
                take_profit=None,
                trailing_atr_mult=None,
            )
        )

    return base


def _make_core_model(core: str, *, random_state: int = 42):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    c = (core or "hgb").strip().lower()

    if c == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        )
    elif c == "mlp":
        from sklearn.neural_network import MLPClassifier

        model = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=random_state,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=800,
            random_state=random_state,
        )

    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", model)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep multiple strategy variants and write bt_* folders")

    p.add_argument("--in-csv", required=True, help="Input OHLCV+feature CSV for one asset")
    p.add_argument("--asset", default=None, help="Asset name for labeling (default: inferred from filename)")
    p.add_argument("--out-dir", default=".", help="Directory to write bt_* folders into")
    p.add_argument("--out-prefix", default="bt_sweep", help="Prefix for output folders (each variant becomes <prefix>_<variant>")

    p.add_argument(
        "--cores",
        default="hgb,rf,mlp",
        help="Comma-separated cores to run: hgb,rf,mlp (default: all)",
    )
    p.add_argument(
        "--only-variants",
        default=None,
        help="Comma-separated variant names to run (e.g., hgb_baseline,hgb_hmm). If omitted, run all variants for selected cores.",
    )

    p.add_argument("--horizon", type=int, default=5, help="Label horizon in bars")
    p.add_argument("--label-threshold", type=float, default=0.002, help="Classification threshold for labels")
    p.add_argument("--train-frac", type=float, default=0.65, help="Train fraction (rest is out-of-sample backtest)")

    p.add_argument("--mode", default="long_short", help="long_short | long_only")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--delay", type=int, default=1)

    args = p.parse_args(argv)

    in_csv = Path(args.in_csv)
    df_raw = _load_csv(in_csv)
    if df_raw.empty:
        raise SystemExit(f"No rows read from {in_csv}")

    asset = str(args.asset) if args.asset else in_csv.stem.split("_", 1)[0]

    # Feature engineering + labels
    X = build_features(df_raw)
    close = pd.to_numeric(df_raw["Close"], errors="coerce")

    spec = LabelSpec(horizon=int(args.horizon), task="classification", threshold=float(args.label_threshold))
    y = make_labels(close, spec)

    # Align and drop NaN labels
    df = pd.DataFrame(index=X.index)
    df["close"] = close
    df["y"] = y

    # Add asset dummies (so models can be compared on same interface as multi-asset models)
    df[f"asset_{asset}"] = 1.0

    # Merge features
    for c in X.columns:
        df[c] = X[c]

    df = df.dropna(subset=["y"]).copy()

    # Train/test split (time-respecting)
    n = len(df)
    min_train = 30
    min_test = 10
    if n < (min_train + min_test):
        # Soften requirements for small datasets while keeping at least a tiny test.
        print(f"Warning: small dataset after labeling (n={n}). Reducing split minimums.")
        min_train = max(5, int(n * 0.6))
        min_test = max(3, n - min_train)

    split = int(n * float(args.train_frac))
    split = max(min_train, split)
    split = min(n - min_test, split)
    if split <= 0 or split >= n:
        raise SystemExit(f"Invalid train/test split: n={n}, split={split}")
    train_idx = df.index[:split]
    test_idx = df.index[split:]

    train_mask = np.asarray(df.index.isin(train_idx), dtype=bool)

    X_all = df.drop(columns=["y"])  # includes close + features + asset dummy
    y_all = df["y"].astype(int)

    # Some models were trained without the explicit close column (close is also a feature in build_features).
    # Keep as-is: close is part of the feature matrix for this sweep.

    # Precompute gates
    ret1 = pd.to_numeric(df["return_1"], errors="coerce") if "return_1" in df.columns else df["close"].pct_change(1)
    ret1 = pd.to_numeric(ret1, errors="coerce").fillna(0.0).astype(float)
    hmm_states = _hmm_states(ret1, train_mask=train_mask, n_states=3)
    hmm_risk_off = _hmm_risk_off_mask(ret1, hmm_states, train_mask=train_mask) if hmm_states is not None else None

    fft_ok = _fft_trade_mask(df["close"], window=128, min_p=10.0, max_p=80.0)
    dmd_ok = _dmd_trade_mask(ret1, window=60, lookback=10, rank=3)

    # ARMA/ARIMA-ish experiment: prewhiten returns with a simple AR(p) fit on the train segment.
    beta_ar5 = _fit_ar_coeffs(ret1, train_mask=train_mask, p=5)
    ret1_ar5 = _ar_residuals(ret1, beta=beta_ar5)

    wavelet_cache: dict[tuple[str, int | None], pd.Series] = {}

    def _get_wavelet_ok(kernel: str | None, ar_p: int | None) -> pd.Series:
        k = str(kernel).strip().lower() if kernel else "dwt_db8"
        p = int(ar_p) if ar_p is not None else None
        key = (k, p)
        if key in wavelet_cache:
            return wavelet_cache[key]
        r = ret1
        if p == 5:
            r = ret1_ar5
        # Shared hyperparams for all wavelet kernels in this sweep.
        wavelet_cache[key] = _wavelet_gate_for_kernel(r, kernel=k, window=128, min_p=10.0, max_p=80.0, thr=2.5)
        return wavelet_cache[key]

    variants = _make_variants()

    cores = [c.strip().lower() for c in str(args.cores).split(",") if c.strip()]
    if not cores:
        cores = ["hgb"]
    variants = [v for v in variants if v.core in set(cores)]

    if args.only_variants:
        only = {x.strip() for x in str(args.only_variants).split(",") if x.strip()}
        variants = [v for v in variants if v.name in only]
        if not variants:
            raise SystemExit(f"No variants matched --only-variants={args.only_variants}")

    out_root = Path(args.out_dir)
    out_dirs: list[Path] = []

    for core in sorted(set(v.core for v in variants)):
        pipe = _make_core_model(core)
        # Fit on train only
        pipe.fit(X_all.loc[train_idx], y_all.loc[train_idx])

        # Predict only on test for backtest; zeros elsewhere
        allow_short = (str(args.mode) or "long_short").strip().lower() not in {"long", "long_only", "long-only"}

        # Precompute proba once where needed
        proba = None
        classes = None
        try:
            proba = pipe.predict_proba(X_all.loc[test_idx])
            classes = getattr(pipe.named_steps.get("model"), "classes_", None)
        except Exception:
            proba = None
            classes = None

        pred = pipe.predict(X_all.loc[test_idx])
        pred = pd.Series(pred, index=test_idx)

        for v in [vv for vv in variants if vv.core == core]:
            sig = pd.Series(0.0, index=df.index)

            if v.use_proba and (proba is not None) and (classes is not None):
                s = _proba_to_signal(np.asarray(classes), np.asarray(proba), enter=float(v.proba_enter), allow_short=allow_short)
                sig.loc[test_idx] = s
            else:
                sig.loc[test_idx] = pd.to_numeric(pred, errors="coerce").fillna(0.0).clip(-1, 1).round(0)

            # Apply gates
            gate = pd.Series(True, index=df.index)
            if v.hmm_gate:
                if hmm_risk_off is None:
                    # hmmlearn not installed or insufficient data -> gate is False (no trading)
                    gate &= False
                else:
                    gate &= (~hmm_risk_off).reindex(df.index).fillna(False)
            if v.fft_gate:
                gate &= fft_ok.reindex(df.index).fillna(False)
            if v.wavelet_gate:
                w_ok = _get_wavelet_ok(v.wavelet_kernel, v.wavelet_ar_p)
                gate &= w_ok.reindex(df.index).fillna(False)
            if v.dmd_gate:
                gate &= dmd_ok.reindex(df.index).fillna(False)

            sig = sig.where(gate, other=0.0)

            # Backtest config
            cfg = BacktestConfig(
                mode=str(args.mode),
                execution_delay=int(args.delay),
                cost_bps=float(args.cost_bps),
                slippage_bps=float(args.slippage_bps),
                initial_equity=1.0,
                vol_target=(float(v.vol_target) if v.vol_target is not None else None),
                vol_lookback=20,
                max_leverage=3.0,
            )

            # Run simulator
            if v.stop_loss is not None or v.take_profit is not None or v.trailing_atr_mult is not None:
                curve, trades = simulate_ohlc(
                    df_raw.loc[df.index],
                    sig,
                    cfg=cfg,
                    mark="open",
                    stop_loss=v.stop_loss,
                    take_profit=v.take_profit,
                    trailing_stop=None,
                    trailing_atr_mult=v.trailing_atr_mult,
                    atr_window=14,
                    stop_priority="conservative",
                )
                m = compute_metrics(curve)
            else:
                curve = compute_equity_curve(df["close"], sig, cfg=cfg)
                m = compute_metrics(curve)
                trades = trades_from_positions(curve["close"], curve["position"])

            tm = compute_trade_metrics(trades)

            out_dir = out_root / f"{args.out_prefix}_{asset}_{v.name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_dirs.append(out_dir)

            curve.to_csv(out_dir / "equity_curve.csv")
            trades.to_csv(out_dir / "trades.csv", index=False)
            _write_equity_plot_html(out_dir / "equity_curve.html", curve, title=f"{asset} - {v.name}")

            summary = {
                "asset": asset,
                "input": str(in_csv),
                "variant": v.name,
                # Align with existing report column name.
                "model": v.core,
                "mode": str(args.mode),
                "task": "classification",
                "signal_source": ("proba" if v.use_proba else "predict"),
                "proba_enter": float(v.proba_enter),
                "cost_bps": float(args.cost_bps),
                "slippage_bps": float(args.slippage_bps),
                "execution_delay": int(args.delay),
                "vol_target": (float(v.vol_target) if v.vol_target is not None else None),
                "stop_loss": v.stop_loss,
                "take_profit": v.take_profit,
                "trailing_atr_mult": v.trailing_atr_mult,
                # Additional components (for reporting)
                "uses_hmm": bool(v.hmm_gate),
                "uses_fft": bool(v.fft_gate),
                "uses_wavelet": bool(v.wavelet_gate),
                "uses_dmd": bool(v.dmd_gate),
                "uses_neural_net": bool(v.core == "mlp"),
                # Wavelet kernel comparison metadata
                "wavelet_kernel": (str(v.wavelet_kernel) if v.wavelet_kernel else None),
                "wavelet_ar_p": (int(v.wavelet_ar_p) if v.wavelet_ar_p is not None else None),
                "metrics": m,
                "trade_metrics": tm,
            }
            with open(out_dir / "backtest_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    print(f"Wrote {len(out_dirs)} runs under prefix: {args.out_prefix}_{asset}_*")
    # Helpful hint if hmmlearn wasn't available
    if hmm_states is None:
        print("Note: hmmlearn not available (HMM variants will produce no trades). Install: pip install hmmlearn")

    # Note: wavelet variants use a Daubechies DWT energy-concentration gate.
    # It's possible (and valid) for the gate to produce no-trade signals if
    # the energy ratio threshold is too strict for the dataset.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
