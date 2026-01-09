"""Dynamic Mode Decomposition (DMD) utilities.

We implement a basic (exact) DMD for a multivariate time series.

Given snapshots X = [x1, x2, ..., x_{m-1}] and X' = [x2, ..., x_m], DMD finds
A ~= X' X^+ and its eigen-decomposition to obtain modes and eigenvalues.

This module is dependency-light (NumPy only).

Research/education only; not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DMDResult:
    eigenvalues: np.ndarray     # (r,)
    modes: np.ndarray           # (n_features, r)
    amplitudes: np.ndarray      # (r,)
    rank: int


def exact_dmd(X: np.ndarray, *, rank: int = 6) -> DMDResult:
    """Compute exact DMD.

    X is expected shape: (n_features, n_time)
    Returns DMD using snapshots X[:, :-1] and X[:, 1:]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_features, n_time)")
    n, m = X.shape
    if m < 3:
        raise ValueError("Need at least 3 time steps for DMD")

    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # SVD of X1
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)

    r = int(max(1, min(rank, U.shape[1])))
    U_r = U[:, :r]
    s_r = s[:r]
    V_r = Vh.conj().T[:, :r]

    # Build A_tilde = U^* X2 V S^{-1}
    S_inv = np.diag(1.0 / np.where(s_r == 0.0, 1e-12, s_r))
    A_tilde = U_r.T @ X2 @ V_r @ S_inv

    eigvals, W = np.linalg.eig(A_tilde)

    # DMD modes
    Phi = X2 @ V_r @ S_inv @ W

    # Least-squares amplitudes to match x1
    x1 = X[:, 0]
    # Solve Phi b ~= x1
    b, *_ = np.linalg.lstsq(Phi, x1, rcond=None)

    return DMDResult(
        eigenvalues=np.asarray(eigvals, dtype=complex),
        modes=np.asarray(Phi, dtype=complex),
        amplitudes=np.asarray(b, dtype=complex),
        rank=r,
    )


def eigenvalue_to_frequency(eigvals: np.ndarray, *, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Convert discrete-time eigenvalues to (growth_rate, frequency).

    For each eigenvalue lambda:
      omega = log(lambda) / dt = growth + i * 2π f

    Returns (growth, freq) where freq is cycles per unit time.
    """
    lam = np.asarray(eigvals, dtype=complex)
    omega = np.log(lam) / float(dt)
    growth = np.real(omega)
    freq = np.imag(omega) / (2.0 * np.pi)
    return growth, freq
