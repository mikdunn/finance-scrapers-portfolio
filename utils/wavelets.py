"""Wavelet utilities.

We use a Daubechies Discrete Wavelet Transform (DWT) gate for time series.

Why DWT (Daubechies) here?
- It's a compactly supported wavelet family (dbN) that often behaves well for
  noisy financial returns.
- Unlike continuous wavelets (e.g., Morlet CWT), we can implement DWT using only
  SciPy's built-in Daubechies filter coefficients, avoiding additional native
  dependencies (PyWavelets is a common pain on Windows + Python 3.13).

"Varying coefficients" implementation:
- We support trying multiple Daubechies orders (db2, db4, db6, db8, ...).
- The final gate is an OR across orders: if *any* dbN sees concentrated energy
  in-band at time t, we allow trading.

Research/education only; not financial advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_returns_from_close(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce").astype(float)
    r = np.diff(np.log(np.clip(s.to_numpy(dtype=float), 1e-12, np.inf)), prepend=np.nan)
    return pd.Series(r, index=s.index, name="ret")


def _sanitize_x(x: pd.Series) -> tuple[np.ndarray, pd.Index]:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    idx = s.index
    arr = s.to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = arr - float(np.mean(arr))
    return arr, idx


def cwt_band_gate_from_returns(
    returns: pd.Series,
    *,
    window: int = 128,
    min_period: float = 10.0,
    max_period: float = 80.0,
    kernel: str = "morlet2",
    n_periods: int = 48,
    min_power_ratio: float = 3.0,
) -> pd.Series:
    """Boolean gate using a CWT power-peakedness test across scales.

    Supported kernels (via SciPy):
    - morlet2 (complex Morlet)
    - ricker (Mexican hat / Ricker)
    """
    x, idx = _sanitize_x(returns)
    n = int(x.size)
    if n == 0:
        return pd.Series(False, index=idx, name="cwt_gate")

    w = int(window)
    w = max(16, w)
    if n < w:
        return pd.Series(False, index=idx, name="cwt_gate")

    min_p = float(min_period)
    max_p = float(max_period)
    if not (np.isfinite(min_p) and np.isfinite(max_p) and max_p > min_p > 0):
        return pd.Series(False, index=idx, name="cwt_gate")

    n_periods = int(max(8, n_periods))
    periods = np.linspace(min_p, max_p, num=n_periods, dtype=float)

    k = (kernel or "morlet2").strip().lower()
    # Pure NumPy CWT via convolution (SciPy's signal.cwt isn't available in some SciPy builds).

    def _conv_same_len(x_: np.ndarray, h_: np.ndarray, *, n_out: int) -> np.ndarray:
        """np.convolve(..., mode='same') but guaranteed to return exactly n_out samples.

        NumPy defines 'same' output length as max(len(x), len(h)), which can exceed
        the input length when the kernel is longer than the series. For our CWT gate
        we always want per-timepoint coefficients aligned to the input series.
        """
        c_ = np.convolve(x_, h_, mode="same")
        if int(c_.size) == int(n_out):
            return c_
        if int(c_.size) < int(n_out):
            # Pad symmetrically (rare) to reach n_out.
            pad = int(n_out) - int(c_.size)
            left = pad // 2
            right = pad - left
            return np.pad(c_, (left, right), mode="constant", constant_values=0.0)

        # Center-crop to n_out.
        start = (int(c_.size) - int(n_out)) // 2
        return c_[start : start + int(n_out)]

    def _morlet2(M: int, s: float, *, w0: float = 6.0) -> np.ndarray:
        t = np.arange(M, dtype=float) - (M - 1) / 2.0
        ts = t / float(s)
        wave = np.exp(1j * w0 * ts) * np.exp(-0.5 * ts * ts)
        # Zero-mean improves numerical stability for DC-heavy series.
        wave = wave - np.mean(wave)
        # Normalize energy so scales are comparable.
        denom = np.sqrt(np.sum(np.abs(wave) ** 2)) + 1e-12
        return (wave / denom).astype(complex)

    def _ricker(M: int, a: float) -> np.ndarray:
        t = np.arange(M, dtype=float) - (M - 1) / 2.0
        ta = t / float(a)
        A = 2.0 / (np.sqrt(3.0 * float(a)) * (np.pi ** 0.25))
        wave = A * (1.0 - ta * ta) * np.exp(-0.5 * ta * ta)
        wave = wave - np.mean(wave)
        denom = np.sqrt(np.sum(wave * wave)) + 1e-12
        return (wave / denom).astype(float)

    if k in {"morl", "morlet", "morlet2"}:
        w0 = 6.0
        widths = np.clip(periods * w0 / (2.0 * float(np.pi)), 1.0, None)
        coeffs = np.zeros((widths.size, n), dtype=complex)
        for i, s in enumerate(widths.tolist()):
            M = int(min(max(31, 10 * float(s)), 801))
            psi = _morlet2(M, float(s), w0=w0)
            c = _conv_same_len(x, np.conjugate(psi[::-1]), n_out=n)
            coeffs[i, :] = c
        power = (np.abs(coeffs) ** 2).astype(float)
    elif k in {"ricker", "mexh", "mexican_hat", "mexican-hat"}:
        widths = np.clip(periods / 2.0, 1.0, None)
        power = np.zeros((widths.size, n), dtype=float)
        for i, a in enumerate(widths.tolist()):
            M = int(min(max(31, 10 * float(a)), 801))
            psi = _ricker(M, float(a))
            c = _conv_same_len(x, psi[::-1], n_out=n)
            power[i, :] = c * c
    else:
        return pd.Series(False, index=idx, name="cwt_gate")

    max_pow = np.nanmax(power, axis=0)
    med_pow = np.nanmedian(power, axis=0)
    ratio = max_pow / (med_pow + 1e-12)
    gate = np.isfinite(ratio) & (ratio >= float(min_power_ratio))
    gate[: w - 1] = False
    return pd.Series(gate.astype(bool), index=idx, name="cwt_gate")


def cwt_band_gate(
    close: pd.Series,
    *,
    window: int = 128,
    min_period: float = 10.0,
    max_period: float = 80.0,
    kernel: str = "morlet2",
    n_periods: int = 48,
    min_power_ratio: float = 3.0,
    use_returns: bool = True,
) -> pd.Series:
    """Convenience wrapper for CWT gating from close prices (or direct series)."""
    x = _to_returns_from_close(close) if use_returns else pd.to_numeric(close, errors="coerce").astype(float)
    return cwt_band_gate_from_returns(
        x,
        window=int(window),
        min_period=float(min_period),
        max_period=float(max_period),
        kernel=str(kernel),
        n_periods=int(n_periods),
        min_power_ratio=float(min_power_ratio),
    )


def _db_filters(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (lowpass, highpass) analysis filters for Daubechies(order)."""
    # Coefficients are the standard Daubechies scaling (low-pass) filter taps.
    # We include a pragmatic subset used by the sweep (db2/4/6/8). This keeps
    # the project pure-Python and avoids SciPy/PyWavelets API/version issues.
    DB_H: dict[int, list[float]] = {
        2: [
            0.4829629131445341,
            0.8365163037378079,
            0.2241438680420134,
            -0.1294095225512603,
        ],
        4: [
            0.2303778133088964,
            0.7148465705529154,
            0.6308807679298587,
            -0.02798376941698385,
            -0.1870348117188811,
            0.030841381835986965,
            0.032883011666982945,
            -0.010597401784997278,
        ],
        6: [
            0.11154074335008017,
            0.4946238903983854,
            0.7511339080210954,
            0.3152503517091982,
            -0.22626469396516913,
            -0.12976686756709563,
            0.09750160558707936,
            0.02752286553001629,
            -0.031582039318031156,
            0.0005538422009938016,
            0.00477725751094551,
            -0.0010773010853084796,
        ],
        8: [
            0.05441584224308161,
            0.3128715909144659,
            0.6756307362972898,
            0.5853546836548691,
            -0.015829105256349306,
            -0.2840155429615824,
            0.0004724845739124,
            0.12874742662048934,
            -0.017369301001807546,
            -0.044088253930794755,
            0.013981027917015516,
            0.008746094047015655,
            -0.00487035299301066,
            -0.000391740373376947,
            0.0006754494059985568,
            -0.00011747678412476954,
        ],
    }

    n = int(order)
    if n not in DB_H:
        raise ValueError(
            f"Unsupported Daubechies order db{n}. Supported: {sorted(DB_H.keys())}."
        )

    h = np.asarray(DB_H[n], dtype=float)
    # Quadrature mirror filter for the corresponding high-pass analysis filter.
    g = np.array([((-1.0) ** k) * h[-1 - k] for k in range(h.size)], dtype=float)
    return h, g


def _dwt_detail_energies(
    x: np.ndarray,
    *,
    order: int,
    max_level: int | None = None,
) -> np.ndarray:
    """Compute per-level detail energy for a 1D DWT (levels j=1..J).

    This is a small analysis-only DWT:
    - Reflect-pad at boundaries
    - Convolve with analysis low/high-pass filters
    - Downsample by 2
    - Record mean square energy of detail coefficients at each level
    """
    x = np.asarray(x, dtype=float)
    if x.size < 32:
        return np.asarray([], dtype=float)

    h, g = _db_filters(order)

    n = int(x.size)
    J = int(np.floor(np.log2(n))) - 1
    if max_level is not None:
        J = min(J, int(max_level))
    if J < 1:
        return np.asarray([], dtype=float)

    pad = int(max(1, h.size - 1))
    a = x.copy()
    energies: list[float] = []

    for _j in range(1, J + 1):
        ap = np.pad(a, pad_width=pad, mode="reflect")
        approx = np.convolve(ap, h, mode="valid")[::2]
        detail = np.convolve(ap, g, mode="valid")[::2]
        energies.append(float(np.mean(detail * detail)) if detail.size else float("nan"))
        a = approx
        if a.size < 8:
            break

    return np.asarray(energies, dtype=float)


def daubechies_dwt_band_gate(
    close: pd.Series,
    *,
    window: int = 128,
    min_period: float = 10.0,
    max_period: float = 80.0,
    orders: tuple[int, ...] = (2, 4, 6, 8),
    min_energy_ratio: float = 2.5,
    use_returns: bool = True,
) -> pd.Series:
    """Boolean trade gate based on Daubechies DWT energy concentration.

    For each time t (after warmup), take trailing window and:
    - compute DWT detail energies across levels
    - select levels where period_j ≈ 2^j candles lies in [min_period, max_period]
    - compute ratio = max_energy / median_energy across selected levels
    - gate True if ratio >= min_energy_ratio

    We try multiple db orders ("varying coefficients") and OR the decision.
    """
    x = _to_returns_from_close(close) if use_returns else pd.to_numeric(close, errors="coerce").astype(float)
    return daubechies_dwt_band_gate_from_returns(
        x,
        window=int(window),
        min_period=float(min_period),
        max_period=float(max_period),
        orders=tuple(int(o) for o in orders),
        min_energy_ratio=float(min_energy_ratio),
    )


def daubechies_dwt_band_gate_from_returns(
    returns: pd.Series,
    *,
    window: int = 128,
    min_period: float = 10.0,
    max_period: float = 80.0,
    orders: tuple[int, ...] = (2, 4, 6, 8),
    min_energy_ratio: float = 2.5,
) -> pd.Series:
    """Same as :func:`daubechies_dwt_band_gate`, but expects returns (or any 1D series)."""
    x, idx = _sanitize_x(returns)
    n = int(x.size)
    if n == 0:
        return pd.Series(False, index=idx, name="dwt_gate")

    w = int(window)
    w = max(32, w)
    if n < w:
        return pd.Series(False, index=idx, name="dwt_gate")

    min_p = float(min_period)
    max_p = float(max_period)
    if not (np.isfinite(min_p) and np.isfinite(max_p) and max_p > min_p > 0):
        return pd.Series(False, index=idx, name="dwt_gate")

    # DWT levels correspond to dyadic scales; approximate "period" as 2^j bars.
    j_min = int(np.ceil(np.log2(min_p)))
    j_max = int(np.floor(np.log2(max_p)))
    if j_max < 1:
        return pd.Series(False, index=idx, name="dwt_gate")

    ords = tuple(int(o) for o in orders if int(o) >= 2) or (4,)
    thr = float(min_energy_ratio)
    out = np.zeros(n, dtype=bool)

    for i in range(w - 1, n):
        seg = x[i - w + 1 : i + 1]
        ok = False
        for o in ords:
            energies = _dwt_detail_energies(seg, order=o)
            if energies.size < 2:
                continue
            js = np.arange(1, energies.size + 1)
            mask = (js >= j_min) & (js <= j_max) & np.isfinite(energies)
            cand = energies[mask]
            if cand.size < 2:
                continue
            max_e = float(np.max(cand))
            med_e = float(np.median(cand))
            ratio = max_e / (med_e + 1e-12)
            if np.isfinite(ratio) and ratio >= thr:
                ok = True
                break
        out[i] = ok

    out[: w - 1] = False
    return pd.Series(out, index=idx, name="dwt_gate")
