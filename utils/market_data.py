"""Market data providers.

Currently uses Yahoo Finance via yfinance as a unified source for:
- Stocks/ETFs
- Many crypto pairs and futures symbols available on Yahoo
- Some FX pairs (Yahoo-specific tickers)

Note: availability depends on Yahoo symbol conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests


def _period_to_start_ts(period: str, now: pd.Timestamp) -> pd.Timestamp | None:
    """Convert a yfinance-like period string into an approximate start timestamp.

    Returns None for 'max' (no trimming).
    """
    # Ensure timezone-naive comparisons with typical price DataFrame indices.
    if getattr(now, 'tzinfo', None) is not None:
        now = now.tz_localize(None)

    p = (period or '').strip().lower()
    if p in {'max', ''}:
        return None
    if p == 'ytd':
        return pd.Timestamp(year=now.year, month=1, day=1)
    if p.endswith('mo'):
        n = int(p[:-2])
        return now - pd.Timedelta(days=30 * n)
    if p.endswith('y'):
        n = int(p[:-1])
        return now - pd.Timedelta(days=365 * n)
    return None


@dataclass(frozen=True)
class OHLCVRequest:
    symbol: str
    period: str = "6mo"          # e.g. 1mo, 3mo, 6mo, 1y, 5y, max
    interval: str = "1d"         # e.g. 1m, 5m, 15m, 1h, 1d
    auto_adjust: bool = False


def fetch_ohlcv(req: OHLCVRequest) -> pd.DataFrame:
    """Fetch OHLCV data for a symbol.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a DatetimeIndex.
    """
    # Provider 1: Yahoo Finance via yfinance
    df = None
    try:
        import yfinance as yf

        t = yf.Ticker(req.symbol)
        df = t.history(period=req.period, interval=req.interval, auto_adjust=req.auto_adjust)
    except Exception:
        df = None

    # Provider 2: Kraken (crypto, no key) for symbols like BTC-USD, ETH-USD
    if df is None or df.empty:
        try:
            df = fetch_ohlcv_kraken(req)
        except Exception:
            df = df

    # Provider 3: Coinbase (crypto, no key) for symbols like BTC-USD, ETH-USD
    # (Some networks block Coinbase; keep as best-effort.)
    if df is None or df.empty:
        try:
            df = fetch_ohlcv_coinbase(req)
        except Exception:
            df = df

    # Provider 4 (fallback): Stooq CSV (daily only; great for stocks/ETFs/FX, some crypto)
    if df is None or df.empty:
        df = fetch_ohlcv_stooq(req)

    if df is None or df.empty:
        raise ValueError(
            f"No OHLCV data returned for symbol '{req.symbol}' (period={req.period}, interval={req.interval})."
        )

    # Standardize column names we rely on
    # yfinance returns: Open High Low Close Volume Dividends Stock Splits
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    return df


def fetch_ohlcv_coinbase(req: OHLCVRequest) -> pd.DataFrame:
    """Fetch OHLCV candles from Coinbase Exchange public API.

    Supports common crypto pairs like BTC-USD, ETH-USD.
    """
    symbol = (req.symbol or '').strip().upper()
    if '-' not in symbol:
        return pd.DataFrame()

    base, quote = symbol.split('-', 1)
    if not (base.isalpha() and quote.isalpha()):
        return pd.DataFrame()

    # Coinbase uses product IDs like BTC-USD
    product_id = f"{base}-{quote}"

    interval = (req.interval or '1d').lower()
    granularity_map = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '6h': 21600,
        '1d': 86400,
    }
    if interval not in granularity_map:
        # If caller requests an unsupported interval, skip.
        return pd.DataFrame()

    granularity = granularity_map[interval]

    # Coinbase endpoint returns max ~300 candles per request. We'll fetch enough for the period.
    now = pd.Timestamp.utcnow()
    start = _period_to_start_ts(req.period, now)
    if start is None:
        # Default to 6 months if max requested (to keep reasonable)
        start = now - pd.Timedelta(days=180)

    # Coinbase expects ISO8601 timestamps
    start_iso = start.to_pydatetime().isoformat() + 'Z'
    end_iso = now.to_pydatetime().isoformat() + 'Z'

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {'start': start_iso, 'end': end_iso, 'granularity': granularity}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()

    data = r.json()
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    # Each row: [time, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=['time', 'Low', 'High', 'Open', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(None)
    df = df.drop(columns=['time']).set_index('Date').sort_index()
    # Ensure numeric
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Trim exactly to start if provided
    if start is not None:
        if getattr(start, 'tzinfo', None) is not None:
            start = start.tz_localize(None)
        df = df[df.index >= start]

    return df


def fetch_ohlcv_kraken(req: OHLCVRequest) -> pd.DataFrame:
    """Fetch OHLCV candles from Kraken public API.

    Kraken pairs use XBT for Bitcoin. Examples:
      - BTC-USD -> XBTUSD
      - ETH-USD -> ETHUSD

    Returns a DataFrame with Open, High, Low, Close, Volume.
    """
    symbol = (req.symbol or '').strip().upper()
    if '-' not in symbol:
        return pd.DataFrame()

    base, quote = symbol.split('-', 1)
    if not (base.isalpha() and quote.isalpha()):
        return pd.DataFrame()

    base_kr = 'XBT' if base == 'BTC' else base
    pair = f"{base_kr}{quote}"

    interval = (req.interval or '1d').lower()
    interval_map = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }
    if interval not in interval_map:
        return pd.DataFrame()

    iv = interval_map[interval]
    url = 'https://api.kraken.com/0/public/OHLC'

    now = pd.Timestamp.utcnow()
    start = _period_to_start_ts(req.period, now)
    since = int(start.timestamp()) if start is not None else None

    params = {'pair': pair, 'interval': iv}
    if since is not None:
        params['since'] = since

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()

    if payload.get('error'):
        return pd.DataFrame()

    result = payload.get('result') or {}
    # Find the actual key for the returned pair (can be like XETHZUSD)
    ohlc_key = None
    for k in result.keys():
        if k == 'last':
            continue
        ohlc_key = k
        break
    if not ohlc_key:
        return pd.DataFrame()

    rows = result.get(ohlc_key) or []
    if not rows:
        return pd.DataFrame()

    # Each row:
    # [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(rows, columns=['time', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Count'])
    df['Date'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert(None)
    df = df.drop(columns=['time', 'VWAP', 'Count']).set_index('Date').sort_index()

    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if start is not None:
        if getattr(start, 'tzinfo', None) is not None:
            start = start.tz_localize(None)
        df = df[df.index >= start]

    return df


def fetch_ohlcv_stooq(req: OHLCVRequest) -> pd.DataFrame:
    """Fetch daily OHLCV from Stooq as a fallback.

    Stooq uses symbols like:
      - aapl.us (US stocks)
      - tsla.us
    - eurusd (FX)
    - btcusd (crypto)

    We map common US tickers automatically.
    """
    if (req.interval or '').lower() != '1d':
        return pd.DataFrame()

    sym_raw = (req.symbol or '').strip()
    sym = sym_raw.lower()
    if not sym:
        return pd.DataFrame()

    # Heuristic mapping:
    # - Yahoo FX tickers: EURUSD=X -> eurusd
    # - Common crypto tickers: BTC-USD -> btcusd
    # - US equities: AAPL -> aapl.us

    stooq_sym: str | None = None

    if sym.endswith('=x') and len(sym) >= 5:
        stooq_sym = sym.replace('=x', '')
    elif '-' in sym and sym.count('-') == 1:
        base, quote = sym.split('-', 1)
        if base.isalpha() and quote.isalpha():
            stooq_sym = f"{base}{quote}"  # BTC-USD -> btcusd
    elif sym.isalpha():
        # Could be FX (eurusd) or crypto (btcusd) already.
        if len(sym) == 6:
            stooq_sym = sym
        else:
            # Assume US equity
            stooq_sym = f"{sym}.us"
    else:
        return pd.DataFrame()

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    r = requests.get(url, timeout=15)
    r.raise_for_status()

    df = pd.read_csv(pd.io.common.StringIO(r.text))
    if df.empty or 'Date' not in df.columns:
        return pd.DataFrame()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date').sort_index()

    # Standardize columns to match yfinance-ish naming
    rename = {
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume',
    }
    df = df.rename(columns=rename)
    # Some Stooq feeds (FX/crypto) do not include Volume.
    keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[keep].copy()
    if 'Volume' not in df.columns:
        df['Volume'] = pd.NA

    # Trim to requested period (approx)
    start = _period_to_start_ts(req.period, pd.Timestamp.utcnow())
    if start is not None:
        df = df[df.index >= start]

    # Stooq is not adjusted; ignore auto_adjust here.
    return df


def fetch_options_expirations(symbol: str) -> list[str]:
    """Return available option expiration dates for a symbol (Yahoo)."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    return list(getattr(t, "options", []) or [])


def fetch_options_chain(symbol: str, expiration: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (calls, puts) options chain for a given expiration date."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    chain = t.option_chain(expiration)
    return chain.calls.copy(), chain.puts.copy()
