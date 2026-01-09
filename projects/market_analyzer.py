"""Market analyzer: candlestick charts + indicators + simple forecasting.

Supports (via Yahoo Finance / yfinance): stocks, ETFs, many futures/forex/crypto symbols
depending on Yahoo ticker formats.

This module produces:
- Interactive candlestick chart HTML
- CSV with computed indicators
- JSON summary with regime / support-resistance / forecast

Note: This is for educational/analysis purposes only; not financial advice.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict

import sys


# Allow running as a script (python projects\market_analyzer.py) without import errors.
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.market_data import OHLCVRequest, fetch_ohlcv
from utils.technical import (
    ichimoku,
    market_regime,
    rsi,
    support_resistance_from_pivots,
    trade_style_heuristic,
)


def _arima_forecast(close: pd.Series, steps: int = 5) -> list[float]:
    """ARIMA forecast of closing price.

    Uses statsmodels if available. Falls back to a naive last-value forecast.
    """
    close = close.dropna().astype(float)
    if len(close) < 50:
        return [float(close.iloc[-1])] * steps if len(close) else []

    try:
        from statsmodels.tsa.arima.model import ARIMA

        # statsmodels is chatty about missing/inferred frequency on market series.
        # It's expected for assets with weekends/holidays, so keep output clean.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Keep it simple and stable: ARIMA(1,1,1)
            model = ARIMA(close, order=(1, 1, 1))
            fitted = model.fit()
            fc = fitted.forecast(steps=steps)
            return [float(x) for x in fc.values]
    except Exception:
        return [float(close.iloc[-1])] * steps


def build_candlestick_figure(df: pd.DataFrame, symbol: str, sr_levels, ich, rsi_series: pd.Series) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f'{symbol} OHLC',
        )
    )

    # Ichimoku lines (overlay)
    fig.add_trace(go.Scatter(x=df.index, y=ich.tenkan_sen, mode='lines', name='Tenkan (9)', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=ich.kijun_sen, mode='lines', name='Kijun (26)', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=ich.senkou_span_a, mode='lines', name='Senkou A', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=ich.senkou_span_b, mode='lines', name='Senkou B', line=dict(width=1)))

    # Support / resistance as horizontal lines
    for lvl in sr_levels.supports:
        fig.add_hline(y=lvl, line_width=1, line_dash='dot', line_color='green', annotation_text=f'S {lvl:.2f}')
    for lvl in sr_levels.resistances:
        fig.add_hline(y=lvl, line_width=1, line_dash='dot', line_color='red', annotation_text=f'R {lvl:.2f}')

    fig.update_layout(
        title=f'{symbol} Candlestick + Ichimoku + S/R',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=750,
    )

    # Add RSI as a secondary figure (simple approach: put it in annotations + separate file via CSV)
    # If you want true multi-panel, we can switch to make_subplots.
    fig.add_annotation(
        x=df.index[-1],
        y=float(df['Close'].iloc[-1]),
        text=f"RSI(14): {float(rsi_series.dropna().iloc[-1]) if rsi_series.dropna().size else float('nan'):.1f}",
        showarrow=True,
        arrowhead=2,
    )

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate candlestick charts + indicators + forecast')
    parser.add_argument('--symbols', default='AAPL,TSLA,SPY', help='Comma-separated symbols')
    parser.add_argument('--period', default='6mo', help='yfinance period (e.g. 1mo, 6mo, 1y, 5y)')
    parser.add_argument('--interval', default='1d', help='yfinance interval (e.g. 1m, 5m, 1h, 1d)')
    parser.add_argument('--out-dir', default='outputs', help='Output directory')
    parser.add_argument('--forecast-steps', type=int, default=5, help='ARIMA forecast horizon (steps)')
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    summaries: list[dict] = []
    failures: list[dict] = []

    for sym in symbols:
        try:
            df = fetch_ohlcv(OHLCVRequest(symbol=sym, period=args.period, interval=args.interval))
        except Exception as e:
            failures.append({'symbol': sym, 'error': str(e)})
            print(f"Warning: skipping {sym} due to data error: {e}")
            continue

        # Indicators
        rsi14 = rsi(df['Close'], 14)
        ichi = ichimoku(df['High'], df['Low'], df['Close'])
        sr = support_resistance_from_pivots(df['High'], df['Low'], window=5)
        regime = market_regime(df['Close']).iloc[-1]
        style = trade_style_heuristic(df['Close'], df['High'], df['Low'], args.interval)

        fc = _arima_forecast(df['Close'], steps=args.forecast_steps)

        # Persist computed frame
        out_df = df.copy()
        out_df['RSI14'] = rsi14
        out_df['Tenkan'] = ichi.tenkan_sen
        out_df['Kijun'] = ichi.kijun_sen
        out_df['SenkouA'] = ichi.senkou_span_a
        out_df['SenkouB'] = ichi.senkou_span_b

        csv_path = os.path.join(args.out_dir, f'{sym}_{args.period}_{args.interval}.csv')
        out_df.to_csv(csv_path)

        fig = build_candlestick_figure(df, sym, sr, ichi, rsi14)
        html_path = os.path.join(args.out_dir, f'{sym}_{args.period}_{args.interval}.html')
        fig.write_html(html_path)

        summary = {
            'symbol': sym,
            'period': args.period,
            'interval': args.interval,
            'last_close': float(df['Close'].iloc[-1]),
            'regime': str(regime),
            'support_levels': [float(x) for x in sr.supports],
            'resistance_levels': [float(x) for x in sr.resistances],
            'trade_style': asdict(style),
            'forecast_close_next': fc,
            'artifacts': {
                'csv': csv_path,
                'html': html_path,
            },
        }
        summaries.append(summary)

        print(f"Wrote {sym}: {html_path} and {csv_path}")

    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'results': summaries, 'failures': failures}, f, indent=2)

    print(f"Wrote summary: {summary_path}")
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
