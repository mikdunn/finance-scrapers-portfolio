# Finance Scrapers Portfolio

Python tools for scraping market data and generating sentiment and visualizations.

## Projects

| Project | Description |
|---|---|
| Sentiment Heatmap | RSS headlines -> VADER sentiment -> Plotly heatmap |
| Multi-Source Collector | Basic collector skeleton (outputs CSV) |
| Market Analyzer | Candlesticks + RSI + Ichimoku + support/resistance + ARIMA forecast |

## Setup

```bash
pip install -r requirements.txt
```

## Run

### Sentiment heatmap

```bash
python main.py --project sentiment_heatmap --source rss --tickers "AAPL,TSLA,SPY"
```

### Market analyzer (crypto + forex)

```bash
python main.py --project market --symbols "BTC-USD,ETH-USD,EURUSD=X,USDJPY=X" --period 6mo --interval 1d --out-dir outputs_fx_crypto
```

### Market analyzer (stocks)

```bash
python main.py --project market --symbols "AAPL,TSLA" --period 6mo --interval 1d --out-dir outputs
```

### Data hub (build datasets, optional charts, optional ML)

Build per-symbol datasets for a universe (S&P 500 / Nasdaq-100) or an explicit symbol list. By default it writes one CSV per symbol plus a `summary.json` manifest.

```bash
python main.py --project hub --universe sp500 --max-symbols 25 --period 1y --interval 1d --out-dir hub_sp500_1y_1d --skip-train
```

If you want the “market analyzer”-style visual artifacts (candlestick HTML + indicator-enriched CSVs), enable chart generation:

```bash
python main.py --project hub --universe nasdaq100 --max-symbols 25 --period 1y --interval 1d --out-dir hub_nasdaq100_1y_1d --generate-charts --skip-train
```

Chart artifacts are written to `<out-dir>/charts/`.

> Note: This project is for educational/analysis purposes only and is not financial advice.
