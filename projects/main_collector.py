import argparse
import os
import sys


# Allow running as a script (python projects\main_collector.py) without import errors.
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from utils.scrapers import fetch_static, parse_ticker_data  # Adapt parse per site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Collect stock data from multiple sources.')
    parser.add_argument('--tickers', default='AAPL,TSLA', help='Comma-separated tickers (default: AAPL,TSLA)')
    parser.add_argument('--out', default='data/multi_source.csv', help='Output CSV path')
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    sources = ['yahoo', 'investing']  # placeholder list; URLs are currently Yahoo-only
    all_data: list[dict] = []

    for ticker in tickers:
        for src in sources:
            url = f"https://finance.yahoo.com/quote/{ticker}"  # TODO: use src to vary URL
            soup = fetch_static(url)
            data = parse_ticker_data(soup, ticker)
            data['source'] = src
            all_data.append(data)

    df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.head())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
