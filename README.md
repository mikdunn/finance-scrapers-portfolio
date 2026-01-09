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

> Note: This project is for educational/analysis purposes only and is not financial advice.
