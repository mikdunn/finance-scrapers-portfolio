"""Portfolio construction / risk allocation utilities.

This project is an analysis + artifact generator, not an execution engine.
These helpers produce *target weights* given historical returns.

Implemented here:
- Cluster-aware allocation: allocate risk/weight across spectral clusters and
  then allocate within each cluster.

Assumptions / notes:
- Uses simple volatility estimates on log-returns.
- Does NOT model transaction costs, slippage, margin, borrow, or liquidity.
- Educational/research use only; not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationResult:
    weights: pd.Series
    diagnostics: dict


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def estimate_volatility(
    returns: pd.DataFrame,
    *,
    lookback: int = 60,
) -> pd.Series:
    """Estimate per-asset volatility using trailing window std of returns."""
    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    n = int(lookback)
    if n <= 2:
        n = min(60, len(returns))

    x = returns.tail(n).astype(float)
    vol = x.std(axis=0, ddof=1)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    return vol


def _normalize_weights(w: pd.Series) -> pd.Series:
    w = w.copy().astype(float)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = float(w.sum())
    if s <= 0.0:
        return w * 0.0
    return w / s


def _apply_bounds(w: pd.Series, *, min_w: float = 0.0, max_w: float | None = None) -> pd.Series:
    w = w.copy().astype(float)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    mn = float(min_w) if min_w is not None else 0.0
    if mn > 0.0:
        w = w.clip(lower=mn)

    if max_w is not None:
        mx = float(max_w)
        if mx > 0.0:
            w = w.clip(upper=mx)

    return w


def cluster_inverse_vol_allocation(
    returns: pd.DataFrame,
    *,
    clusters: pd.Series | dict[str, int] | None,
    lookback: int = 60,
    cluster_budget: str = "inverse_vol",
    within_cluster: str = "inverse_vol",
    min_weight: float = 0.0,
    max_weight: float | None = None,
    eps: float = 1e-12,
) -> AllocationResult:
    """Allocate weights across clusters then within each cluster.

    Parameters
    - returns: DataFrame [t x assets] of returns (e.g., log returns)
    - clusters: mapping asset->cluster_id (ints). If None, all assets in cluster 0.
    - cluster_budget: 'equal' | 'inverse_vol'
    - within_cluster: 'equal' | 'inverse_vol'

    Returns
    - AllocationResult(weights: Series indexed by symbol)
    """
    if returns is None or returns.empty:
        return AllocationResult(pd.Series(dtype=float), {"reason": "empty_returns"})

    cols = list(returns.columns)
    if not cols:
        return AllocationResult(pd.Series(dtype=float), {"reason": "no_assets"})

    if clusters is None:
        cl = pd.Series({c: 0 for c in cols}, dtype=int)
    elif isinstance(clusters, dict):
        cl = pd.Series({c: int(clusters.get(c, 0)) for c in cols}, dtype=int)
    else:
        cl = pd.Series(clusters).copy()
        # ensure it covers requested columns
        cl = cl.reindex(cols).fillna(0).astype(int)

    # Per-asset vol
    vol = estimate_volatility(returns, lookback=int(lookback))
    vol = vol.reindex(cols)

    # Cluster returns proxy: equal-weight within cluster
    cluster_ids = sorted({int(x) for x in cl.values.tolist()})
    cluster_ret = {}
    cluster_assets = {}
    for cid in cluster_ids:
        assets = [a for a in cols if int(cl.loc[a]) == cid]
        cluster_assets[cid] = assets
        if not assets:
            continue
        cluster_ret[cid] = returns[assets].mean(axis=1)

    cluster_ret_df = pd.DataFrame(cluster_ret)
    if cluster_ret_df.empty:
        # fallback: treat as one cluster
        cluster_ret_df = pd.DataFrame({0: returns.mean(axis=1)})
        cluster_assets = {0: cols}

    # Cluster vol
    cvol = estimate_volatility(cluster_ret_df, lookback=int(lookback))

    cb = (cluster_budget or "inverse_vol").strip().lower()
    if cb in {"equal", "ew"}:
        c_w = pd.Series({cid: 1.0 for cid in cluster_ret_df.columns})
    else:
        c_w = 1.0 / (cvol.replace(0.0, np.nan) + eps)

    c_w = _normalize_weights(c_w)

    wc = (within_cluster or "inverse_vol").strip().lower()

    w = pd.Series(0.0, index=cols, dtype=float)
    for cid in cluster_ret_df.columns:
        assets = cluster_assets.get(int(cid), [])
        if not assets:
            continue

        if wc in {"equal", "ew"}:
            w_in = pd.Series({a: 1.0 for a in assets}, dtype=float)
        else:
            v = vol.reindex(assets)
            w_in = 1.0 / (v.replace(0.0, np.nan) + eps)

        w_in = _normalize_weights(w_in)
        w.loc[assets] = float(c_w.loc[cid]) * w_in

    # Apply bounds then renormalize.
    w = _apply_bounds(w, min_w=float(min_weight or 0.0), max_w=max_weight)
    w = _normalize_weights(w)

    # Simple diagnostics (diagonal covariance approximation)
    vol2 = (vol.fillna(0.0) ** 2).reindex(cols).fillna(0.0)
    port_var_diag = float((w**2 * vol2).sum())
    rc_diag = (w**2 * vol2) / (port_var_diag + eps) if port_var_diag > 0 else (w * 0.0)

    diag = {
        "lookback": int(lookback),
        "cluster_budget": cb,
        "within_cluster": wc,
        "clusters": {k: int(v) for k, v in cl.to_dict().items()},
        "cluster_weights": {int(k): _safe_float(v) for k, v in c_w.to_dict().items()},
        "asset_vol": {k: _safe_float(v) for k, v in vol.to_dict().items()},
        "risk_contrib_diag": {k: _safe_float(v) for k, v in rc_diag.to_dict().items()},
    }

    return AllocationResult(weights=w, diagnostics=diag)
