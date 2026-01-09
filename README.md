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

### Spectral clustering (Fiedler vector) + Dynamic Mode Decomposition (DMD)

The hub can also compute multi-asset structure from the set of symbols you fetch:

- **Spectral graph clustering**: builds a similarity graph from aligned log-returns, computes the graph Laplacian, and uses the **Fiedler vector** (2nd-smallest eigenvector) for a simple 2-way clustering.
- **DMD**: runs **Dynamic Mode Decomposition** on the aligned multi-asset log-returns to extract dominant modes/eigenvalues.

Example:

```bash
python main.py --project hub --symbols "ETH-USD,BTC-USD,EURUSD=X" --period 6mo --interval 1d --out-dir hub_spectral --skip-train --spectral --spectral-k 2 --dmd --dmd-rank 3
```

Artifacts written into `<out-dir>/`:

- `spectral_clusters.csv` and `spectral_graph.json`
- `dmd_eigs.csv` and `dmd_summary.json`

> Note: This project is for educational/analysis purposes only and is not financial advice.
