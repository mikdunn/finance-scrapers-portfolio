"""Spectral graph theory utilities for market data.

Primary use here:
- Build an asset-to-asset similarity graph (e.g., correlation on returns)
- Compute the Fiedler vector (2nd-smallest Laplacian eigenvector)
- Use it for a simple 2-way clustering / regime segmentation

This is intentionally dependency-light (NumPy/Pandas only).

Research/education only; not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectralClusteringResult:
    symbols: list[str]
    similarity: np.ndarray          # (n,n)
    adjacency: np.ndarray           # (n,n)
    eigenvalues: np.ndarray         # (n,)
    fiedler_value: float | None
    fiedler_vector: np.ndarray | None
    labels: np.ndarray | None       # (n,) ints


def similarity_from_returns(
    returns: pd.DataFrame,
    *,
    method: str = "pearson",
    use_abs: bool = False,
) -> pd.DataFrame:
    """Compute an asset similarity matrix from returns.

    Returns a DataFrame S with values in [0, 1].

    - method: pearson|spearman
    - use_abs: if True, uses |corr| (treat + and - correlation as similar)
    """
    if returns is None or returns.empty:
        return pd.DataFrame()

    m = (method or "pearson").strip().lower()
    corr = returns.corr(method="spearman" if m.startswith("spear") else "pearson")
    corr = corr.fillna(0.0)

    if use_abs:
        corr = corr.abs()

    # Map correlation [-1,1] -> similarity [0,1]
    sim = (corr + 1.0) / 2.0
    sim.values[np.diag_indices_from(sim.values)] = 0.0
    return sim


def knn_adjacency(similarity: pd.DataFrame, *, k: int = 5, symmetric: bool = True) -> pd.DataFrame:
    """Create a kNN adjacency matrix from a similarity matrix.

    Keeps the top-k similarities per node.
    """
    if similarity is None or similarity.empty:
        return pd.DataFrame()

    k = int(k)
    n = similarity.shape[0]
    if k < 1:
        k = 1
    if k >= n:
        # fully-connected (minus diagonal)
        adj = similarity.copy()
        adj.values[np.diag_indices_from(adj.values)] = 0.0
        return adj

    S = similarity.to_numpy(dtype=float)
    A = np.zeros_like(S)

    for i in range(n):
        row = S[i].copy()
        row[i] = 0.0
        idx = np.argsort(row)[::-1][:k]
        A[i, idx] = row[idx]

    if symmetric:
        A = np.maximum(A, A.T)

    return pd.DataFrame(A, index=similarity.index, columns=similarity.columns)


def laplacian(adjacency: pd.DataFrame, *, normalized: bool = True) -> np.ndarray:
    """Compute (normalized) graph Laplacian."""
    if adjacency is None or adjacency.empty:
        return np.zeros((0, 0), dtype=float)

    A = adjacency.to_numpy(dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)
    D = np.diag(deg)

    if not normalized:
        return D - A

    # L = I - D^{-1/2} A D^{-1/2}
    with np.errstate(divide="ignore"):
        inv_sqrt = 1.0 / np.sqrt(deg)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    Dm = np.diag(inv_sqrt)
    I = np.eye(A.shape[0])
    return I - Dm @ A @ Dm


def fiedler_vector(L: np.ndarray, *, eps: float = 1e-10) -> tuple[float | None, np.ndarray | None, np.ndarray]:
    """Return (fiedler_value, fiedler_vector, eigenvalues).

    Uses dense eigen-decomposition (ok for up to a few thousand assets).
    """
    if L.size == 0:
        return None, None, np.asarray([])

    # L is symmetric (normalized laplacian)
    evals, evecs = np.linalg.eigh(L)
    evals = np.asarray(evals, dtype=float)

    # Find the first eigenvalue > eps (skip the zero eigenvalue)
    idx = None
    for i, v in enumerate(evals):
        if v > eps:
            idx = i
            break

    if idx is None or idx >= evecs.shape[1]:
        return None, None, evals

    fv = np.asarray(evecs[:, idx], dtype=float)
    return float(evals[idx]), fv, evals


def bipartition_labels(fiedler: np.ndarray, *, method: str = "median") -> np.ndarray:
    """Turn a Fiedler vector into 2 cluster labels."""
    v = np.asarray(fiedler, dtype=float)
    m = (method or "median").strip().lower()

    if m in {"sign", "zero"}:
        return (v >= 0.0).astype(int)

    thr = float(np.nanmedian(v))
    return (v >= thr).astype(int)


def spectral_cluster_assets(
    returns: pd.DataFrame,
    *,
    k: int = 5,
    normalized: bool = True,
    corr_method: str = "pearson",
    use_abs_corr: bool = False,
    label_method: str = "median",
) -> SpectralClusteringResult:
    """Compute a simple 2-way spectral clustering from returns."""
    if returns is None or returns.empty:
        return SpectralClusteringResult([], np.zeros((0, 0)), np.zeros((0, 0)), np.asarray([]), None, None, None)

    returns = returns.dropna(axis=1, how="all").copy()
    if returns.shape[1] < 3:
        # too few assets for meaningful clustering
        syms = list(returns.columns)
        n = len(syms)
        return SpectralClusteringResult(syms, np.zeros((n, n)), np.zeros((n, n)), np.asarray([]), None, None, None)

    sim = similarity_from_returns(returns, method=corr_method, use_abs=use_abs_corr)
    adj = knn_adjacency(sim, k=k, symmetric=True)
    L = laplacian(adj, normalized=normalized)

    fval, fvec, evals = fiedler_vector(L)
    labels = None
    if fvec is not None:
        labels = bipartition_labels(fvec, method=label_method)

    return SpectralClusteringResult(
        symbols=list(adj.index),
        similarity=sim.to_numpy(dtype=float),
        adjacency=adj.to_numpy(dtype=float),
        eigenvalues=np.asarray(evals, dtype=float),
        fiedler_value=fval,
        fiedler_vector=fvec,
        labels=labels,
    )
