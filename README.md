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

### Portfolio risk allocation across clusters (analysis-only)

If you enable spectral clustering, the hub can also output a **cluster-aware target weight vector**:

- Allocate weight **across clusters** (default: inverse-vol)
- Allocate weight **within each cluster** (default: inverse-vol)

Example (allocate over the full fetched universe):

```bash
python main.py --project hub --symbols "ETH-USD,BTC-USD,EURUSD=X" --period 6mo --interval 1d --out-dir hub_alloc \
	--skip-train --spectral --spectral-k 2 --allocate-portfolio --alloc-lookback 60
```

Artifacts written into `<out-dir>/`:

- `portfolio_weights.csv`
- `portfolio_allocation.json`

If you’re also using chart selection, you can allocate only across the selected set:

```bash
python main.py --project hub --universe sp500 --max-symbols 50 --period 1y --interval 1d --out-dir hub_sp500_selected_alloc \
	--generate-charts --select-charts --min-conditions 3 --skip-train \
	--spectral --allocate-portfolio --alloc-universe selected
```

## What the analysis tools “mean” (and when to use them)

This repo focuses on **market description + artifacts**. It does **not** place trades or integrate with brokers.

### Candlestick + indicator artifacts (Market Analyzer / Hub charts)

- **Candlesticks**: price action context (trend, volatility, gaps).
- **Support/Resistance (pivot-based)**: rough areas where price previously reacted.
- **Ichimoku**: trend + momentum + “structure” filter; tends to behave best on **trend-friendly** timeframes (e.g., 1h–1d) and liquid markets.
- **RSI(14)**: momentum / mean-reversion proxy; most useful as a *filter* (avoid buying when already overbought).

Best fits:
- **Stocks/ETFs**: ✅ (daily/weekly especially)
- **Futures**: ✅ (but be careful with continuous-contract data quirks)
- **FX/Crypto**: ✅ (often smoother on 1h–1d; beware weekend effects)
- **Options**: ⚠️ usually apply these to the **underlying** chart; this repo does not model greeks/IV/chain microstructure.

### Fibonacci proximity (Hub chart selection condition)

Simple “is price near a common retracement level?” heuristic. Most useful when you already believe you’re in a trending regime and want pullback entries.

Best fits:
- Trending markets/timeframes; less reliable in choppy ranges.

### News (RSS) condition

Fast “is it in the headlines?” proxy for equities. This can help explain *why* volatility expanded, but it’s not a timing tool.

Best fits:
- **Equities** ✅
- **Macro/FX/Crypto** ⚠️ may need different queries/sources

### Fourier dominant cycles (FFT)

Extracts dominant periodicities (in candles) from recent history. Useful for detecting “rhythm” in mean-reverting or range-ish markets.

Best fits:
- Sideways/ranging regimes; less reliable in strong trends or when volatility regime shifts.

### Spectral similarity graphs + Fiedler clustering

Builds an asset similarity graph from aligned returns, then uses the Fiedler vector for a simple 2-way clustering.

What it’s good for:
- **Diversification awareness** (avoid accidentally holding many near-duplicates)
- **Grouping** assets that move together

Best fits:
- Multi-asset universes (sectors, baskets, crypto majors). Needs enough overlapping history.

### Dynamic Mode Decomposition (DMD)

Decomposes the multivariate return dynamics into modes with growth/decay and oscillation frequency.

What it’s good for:
- “System-level” structure across a basket (shared modes)
- Exploratory diagnostics (dominant behaviors)

Best fits:
- Multi-asset baskets; interpret with care (sensitive to noise and lookback choices).

### ML trainer

Trains simple supervised models using engineered features from the analyzer CSVs (walk-forward CV supported).

Best fits:
- Hypothesis testing and ranking features; not a guaranteed edge.

> Note: This project is for educational/analysis purposes only and is not financial advice.
