import argparse
import pandas as pd
from utils.scrapers import fetch_static, parse_ticker_data  # Adapt parse per site

tickers = ['AAPL', 'TSLA']
sources = ['yahoo', 'investing']  # URLs predefined
all_data = []

for ticker in tickers:
    for src in sources:
        url = f"https://finance.yahoo.com/quote/{ticker}"  # etc.
        soup = fetch_static(url)
        data = parse_ticker_data(soup, ticker)
        data['source'] = src
        all_data.append(data)

df = pd.DataFrame(all_data)
df.to_csv('data/multi_source.csv', index=False)
print(df.head())
