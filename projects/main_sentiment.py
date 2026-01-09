import argparse
import os
import sys


# Allow running as a script (python projects\main_sentiment.py) without import errors.
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import re
import xml.etree.ElementTree as ET

from utils.scrapers import fetch_dynamic, fetch_static
from utils.sentiment import score_text


def _fetch_rss_titles(url: str) -> list[str]:
    """Fetch RSS/Atom feed and return item titles.

    Uses stdlib XML parsing to avoid adding new dependencies.
    """
    import requests

    r = requests.get(url, timeout=15, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    r.raise_for_status()
    xml_bytes = r.content

    root = ET.fromstring(xml_bytes)
    titles: list[str] = []

    # RSS: /rss/channel/item/title
    for item in root.findall('.//item'):
        title_el = item.find('title')
        if title_el is not None and title_el.text:
            titles.append(title_el.text.strip())

    # Atom: /feed/entry/title
    if not titles:
        for title_el in root.findall('.//{http://www.w3.org/2005/Atom}title'):
            if title_el.text:
                titles.append(title_el.text.strip())

    # Normalize whitespace and drop obvious duplicates
    cleaned: list[str] = []
    seen = set()
    for t in titles:
        t = re.sub(r'\s+', ' ', t).strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)
    return cleaned


def headlines_from_rss(ticker: str, *, limit: int = 10) -> list[str]:
    """Option A: Get headlines without Selenium (RSS-based, very reliable)."""
    ticker = ticker.strip().upper()

    # Google News RSS search (usually accessible and static)
    google_url = (
        'https://news.google.com/rss/search?'
        f'q={ticker}%20stock&hl=en-US&gl=US&ceid=US:en'
    )
    try:
        titles = _fetch_rss_titles(google_url)
        if titles:
            return titles[:limit]
    except Exception as e:
        print(f"Warning: Google News RSS fetch failed for {ticker}: {e}")

    # Yahoo RSS fallback (sometimes blocked/changed)
    yahoo_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    try:
        titles = _fetch_rss_titles(yahoo_url)
        return titles[:limit]
    except Exception as e:
        print(f"Warning: Yahoo RSS fetch failed for {ticker}: {e}")
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Scrape news headlines and build a sentiment heatmap.')
    parser.add_argument('--tickers', default='AAPL,TSLA,SPY', help='Comma-separated tickers (default: AAPL,TSLA,SPY)')
    parser.add_argument('--headless', action='store_true', help='Run browser automation headless (when Selenium fallback is used)')
    parser.add_argument('--source', default='rss', help='Headline source: rss | yahoo_static | selenium')
    parser.add_argument('--browser', default='edge', help="Selenium browser: edge | chrome (default: edge)")
    parser.add_argument('--out-csv', default='data/sentiments.csv', help='Output CSV path')
    parser.add_argument('--out-html', default='sentiment_heatmap.html', help='Output Plotly HTML path')
    args = parser.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    news_data: list[dict] = []

    for ticker in tickers:
        source = (args.source or 'rss').strip().lower()
        headlines: list[str] = []

        if source == 'rss':
            headlines = headlines_from_rss(ticker, limit=10)
        elif source in {'yahoo_static', 'static'}:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            try:
                soup = fetch_static(url)
                headlines = [h.get_text(strip=True) for h in soup.select('h3') if h.get_text(strip=True)]
            except Exception as e:
                print(f"Warning: Yahoo static fetch failed for {ticker}: {e}")
        elif source == 'selenium':
            url = f"https://finance.yahoo.com/quote/{ticker}"
            try:
                soup = fetch_dynamic(url, headless=args.headless, browser=args.browser)
                headlines = [h.get_text(strip=True) for h in soup.select('h3') if h.get_text(strip=True)]
            except Exception as e:
                print(f"Warning: Selenium fetch failed for {ticker}: {e}")
        else:
            print(f"Warning: unknown source '{args.source}', using rss")
            headlines = headlines_from_rss(ticker, limit=10)

        if not headlines:
            print(f"Warning: no headlines found for {ticker} (source={args.source}).")
        for hl in headlines[:10]:
            news_data.append({'ticker': ticker, 'headline': hl, 'sentiment': score_text(hl)})

    df = pd.DataFrame(news_data)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    if not df.empty:
        pivot = (
            df.pivot_table(values='sentiment', index='ticker', columns='headline', aggfunc='mean')
            .fillna(0)
        )
        import plotly.express as px

        fig = px.imshow(pivot, title='Stock Sentiment Heatmap', aspect='auto')
        fig.write_html(args.out_html)
        # Don't force a GUI pop-up in automated/headless use; HTML output is the artifact.
        if not args.headless:
            fig.show()
    else:
        print('No sentiment rows produced; skipping heatmap generation.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
