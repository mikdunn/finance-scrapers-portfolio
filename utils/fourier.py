"""Fourier / dominant-cycle utilities.

This module provides lightweight FFT-based features for market time series.

Notes:
- We assume candles are evenly spaced (1d, 1h, etc.). The returned periods are
  expressed in *candles*.
- FFT features are heuristic and should be validated per asset/timeframe.

Research/education only; not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DominantCycle:
    k: int
    period: float          # in candles
    frequency: float       # cycles per candle
    power: float
    amplitude: float
    phase: float           # radians


def _detrend(x: np.ndarray, method: str = "linear") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    m = (method or "linear").strip().lower()
    if m in {"none", "off"}:
        return x

    if m in {"mean", "constant"}:
        return x - np.nanmean(x)

    # default: linear detrend
    t = np.arange(x.size, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return x - np.nanmean(x)

    # Fit y = a*t + b
    a, b = np.polyfit(t[mask], x[mask], 1)
    return x - (a * t + b)


def dominant_cycles_fft(
    close: pd.Series,
    *,
    window: int = 256,
    top_k: int = 3,
    min_period: float = 5.0,
    max_period: float | None = None,
    detrend: str = "linear",
    taper: str = "hann",
) -> list[DominantCycle]:
    """Compute dominant cycles using FFT power spectrum on the trailing window.

    Returns up to `top_k` cycles sorted by descending power.
    """
    if close is None:
        return []

    s = pd.to_numeric(close, errors="coerce").astype(float)
    s = s.dropna()
    if s.size < 32:
        return []

    n = int(min(window, s.size))
    x = s.iloc[-n:].to_numpy(dtype=float)

    # Detrend
    x = _detrend(x, detrend)

    # Taper to reduce spectral leakage
    tp = (taper or "hann").strip().lower()
    if tp in {"hann", "hanning"}:
        w = np.hanning(n)
        xw = x * w
    elif tp in {"hamm", "hamming"}:
        w = np.hamming(n)
        xw = x * w
    else:
        xw = x

    # Real FFT
    fft = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0)

    # Power spectrum (ignore DC component at 0 Hz)
    power = (np.abs(fft) ** 2)
    if power.size <= 1:
        return []

    freqs = freqs[1:]
    power = power[1:]
    fft_nodc = fft[1:]

    # Convert to periods in candles
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / freqs

    if max_period is None:
        # Nyquist-ish guard: periods can be as large as ~n
        max_period = float(n)

    mask = np.isfinite(periods) & (periods >= float(min_period)) & (periods <= float(max_period))
    if not np.any(mask):
        return []

    periods_f = periods[mask]
    freqs_f = freqs[mask]
    power_f = power[mask]
    fft_f = fft_nodc[mask]

    # pick top_k by power
    k = int(max(1, top_k))
    top_idx = np.argsort(power_f)[::-1][:k]

    cycles: list[DominantCycle] = []
    for i, idx in enumerate(top_idx, start=1):
        f = float(freqs_f[idx])
        p = float(periods_f[idx])
        pw = float(power_f[idx])
        comp = fft_f[idx]
        # amplitude heuristic: scaled magnitude
        amp = float(2.0 * np.abs(comp) / n)
        ph = float(np.angle(comp))
        cycles.append(DominantCycle(k=i, period=p, frequency=f, power=pw, amplitude=amp, phase=ph))

    return cycles


def rolling_dominant_period(
    close: pd.Series,
    *,
    window: int = 128,
    min_period: float = 5.0,
    max_period: float | None = None,
    detrend: str = "linear",
) -> pd.Series:
    """Rolling dominant period (candles) from the trailing FFT window.

    This returns a Series aligned to `close.index`, with NaN until enough history.
    """
    s = pd.to_numeric(close, errors="coerce").astype(float)
    n = len(s)
    w = int(window)
    if n == 0 or w < 16:
        return pd.Series(index=s.index, dtype=float)

    out = np.full(n, np.nan, dtype=float)
    for i in range(w - 1, n):
        seg = s.iloc[i - w + 1 : i + 1]
        cycles = dominant_cycles_fft(
            seg,
            window=w,
            top_k=1,
            min_period=min_period,
            max_period=max_period,
            detrend=detrend,
        )
        if cycles:
            out[i] = float(cycles[0].period)

    return pd.Series(out, index=s.index, name=f"FFTPeriod_w{w}")
