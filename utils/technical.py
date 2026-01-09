"""Technical analysis helpers.

These functions are intended for analysis/visualization. They do not constitute
financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's EMA: alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


@dataclass(frozen=True)
class Ichimoku:
    tenkan_sen: pd.Series
    kijun_sen: pd.Series
    senkou_span_a: pd.Series
    senkou_span_b: pd.Series
    chikou_span: pd.Series


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
             tenkan: int = 9, kijun: int = 26, senkou_b: int = 52, displacement: int = 26) -> Ichimoku:
    """Compute Ichimoku Cloud components.

    Returns series aligned to the original index. Senkou spans are shifted forward
    by `displacement`; chikou is shifted backward.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2.0).shift(displacement)

    chikou_span = close.shift(-displacement)

    return Ichimoku(
        tenkan_sen=tenkan_sen,
        kijun_sen=kijun_sen,
        senkou_span_a=senkou_span_a,
        senkou_span_b=senkou_span_b,
        chikou_span=chikou_span,
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).rolling(period).mean()


@dataclass(frozen=True)
class SupportResistance:
    supports: list[float]
    resistances: list[float]


def support_resistance_from_pivots(high: pd.Series, low: pd.Series, window: int = 5, max_levels: int = 6,
                                  tolerance: float = 0.003) -> SupportResistance:
    """Detect support/resistance levels from local pivot highs/lows.

    - `window` controls pivot detection (larger => fewer pivots).
    - `tolerance` merges nearby levels (fractional).

    Returns approximate horizontal levels.
    """
    h = high.astype(float)
    l = low.astype(float)

    piv_hi = []
    piv_lo = []

    for i in range(window, len(h) - window):
        hi = h.iloc[i]
        lo = l.iloc[i]
        if np.isfinite(hi) and hi == h.iloc[i - window:i + window + 1].max():
            piv_hi.append(float(hi))
        if np.isfinite(lo) and lo == l.iloc[i - window:i + window + 1].min():
            piv_lo.append(float(lo))

    def _merge(levels: list[float]) -> list[float]:
        levels = sorted(levels)
        merged: list[float] = []
        for x in levels:
            if not merged:
                merged.append(x)
                continue
            if abs(x - merged[-1]) / max(1e-9, merged[-1]) <= tolerance:
                merged[-1] = (merged[-1] + x) / 2.0
            else:
                merged.append(x)
        return merged

    supports = _merge(piv_lo)[-max_levels:]
    resistances = _merge(piv_hi)[:max_levels]

    # Heuristic: supports should be below resistances generally; keep as-is but sorted.
    return SupportResistance(supports=sorted(supports), resistances=sorted(resistances))


def market_regime(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Simple bull/bear regime based on moving averages.

    Returns a series of strings: 'bull', 'bear', or 'neutral'.
    """
    c = close.astype(float)
    f = sma(c, fast)
    s = sma(c, slow)

    out = pd.Series(index=c.index, dtype='object')
    out[(f > s)] = 'bull'
    out[(f < s)] = 'bear'
    out[(f.isna()) | (s.isna())] = 'neutral'
    return out


@dataclass(frozen=True)
class TradeStyle:
    label: str  # long | swing | day | scalp
    rationale: str


def trade_style_heuristic(close: pd.Series, high: pd.Series, low: pd.Series, interval: str) -> TradeStyle:
    """Heuristic classification of trade "style" based on volatility and timeframe.

    This is NOT a trading recommendation; it's just a rule-of-thumb label.
    """
    c = close.astype(float)
    a = atr(high, low, c, period=14)
    vol = (a / c).iloc[-1] if len(c) else np.nan

    if interval.endswith('m') or interval.endswith('h'):
        # Intraday series
        if np.isfinite(vol) and vol >= 0.015:
            return TradeStyle('day', f'higher intraday volatility (ATR/Close≈{vol:.2%})')
        return TradeStyle('scalp', f'lower intraday volatility (ATR/Close≈{vol:.2%})')

    # Daily+ series
    if np.isfinite(vol) and vol >= 0.04:
        return TradeStyle('swing', f'higher daily volatility (ATR/Close≈{vol:.2%})')
    return TradeStyle('long', f'lower daily volatility (ATR/Close≈{vol:.2%})')
