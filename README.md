# Finance Scrapers Portfolio

Python tools for **market analysis + artifact generation**:

- pull OHLCV data (best-effort, with fallbacks)
- compute indicators/features
- generate Plotly HTML charts + CSV/JSON artifacts
- run optional ML experiments (walk-forward validation)
- run multi-asset structure tools (FFT cycles, spectral clustering, DMD)
- compute **cluster-aware portfolio target weights** (analysis-only)

> This repo is for education/research. It is **not financial advice** and it is **not** a broker execution engine.

## Quickstart

1) Create/activate a Python environment and install deps:

```bash
pip install -r requirements.txt
```

2) Run a project via the unified entrypoint:

```bash
python main.py --project market --symbols "AAPL,MSFT" --period 6mo --interval 1d --out-dir outputs
```

Outputs are written to the folder you pass (HTML charts, CSVs, plus a `summary.json` manifest).

## Repository layout

- `main.py`: unified CLI router (`--project market|ml|hub|sentiment_heatmap|collector`)
- `projects/`
	- `market_analyzer.py`: per-symbol chart + indicators + forecast + FFT cycles
	- `ml_train.py`: supervised model training + walk-forward CV + feature importance
	- `data_hub_train.py`: multi-symbol dataset builder + optional charts + selection + spectral/DMD + allocation
	- `main_sentiment.py`: RSS sentiment heatmap (VADER)
	- `main_collector.py`: basic multi-source scraping skeleton
- `utils/`: data fetchers + indicators + spectral/DMD/FFT + portfolio allocation

## Configuration (API keys)

This repo supports optional external APIs. Keys are read from environment variables and the repo includes a `.env` file template.

Current `.env` keys:

- `FRED_API_KEY` (macro features)
- `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `ALPACA_BASE_URL` (present for future expansion; **execution is not implemented** in this repo)

Keep `.env` private and never commit real secrets.

## Projects and how to run them

### 1) Sentiment heatmap (RSS → VADER)

Generates a Plotly heatmap HTML summarizing headline sentiment for tickers.

```bash
python main.py --project sentiment_heatmap --source rss --tickers "AAPL,TSLA,SPY"
```

Best for: equities context / headline tone (not a timing signal).

### 2) Market analyzer (per-symbol charts + indicators)

Produces per-symbol artifacts:

- `SYMBOL_period_interval.html` (interactive candlestick)
- `SYMBOL_period_interval.csv` (OHLCV + indicators/features)
- `summary.json`

Examples:

```bash
python main.py --project market --symbols "BTC-USD,ETH-USD,EURUSD=X" --period 6mo --interval 1d --out-dir outputs_fx_crypto
python main.py --project market --symbols "AAPL,TSLA" --period 6mo --interval 1d --out-dir outputs_stocks
```

Indicators/features included (core):

- RSI(14)
- Ichimoku (Tenkan/Kijun/Senkou A/B)
- Pivot-based support/resistance levels
- Market regime heuristic
- Trade style heuristic
- Optional ARIMA forecast (best-effort)
- FFT dominant cycles (see below)

### 3) ML training (supervised experiments)

Consumes CSV(s) produced by the market analyzer.

Single-asset:

```bash
python main.py --project ml --in-csv outputs_stocks/AAPL_6mo_1d.csv --model hgb --task classification --horizon 5 --threshold 0.003 --cv walkforward --n-splits 6 --test-window 20 --importance permutation --top-features 25 --out-dir ml_outputs_aapl
```

Multi-asset training (combines all CSVs in a directory and adds asset one-hot columns):

```bash
python main.py --project ml --in-dir outputs_stocks --multi-asset --model hgb --task classification --horizon 5 --threshold 0.003 --cv walkforward --n-splits 6 --test-window 20 --importance permutation --top-features 25 --out-dir ml_outputs_multi
```

ML artifacts:

- `model.joblib`
- `metrics.json`
- `feature_importance.csv` and `feature_importance.html` (when applicable)
- `cv_metrics.csv` and `cv_plot.html` (walk-forward)

### 4) Data hub (multi-symbol datasets + optional charts + advanced analysis)

The hub is the “do a bunch of symbols and write a manifest” workflow.

It can:

- build per-symbol dataset CSVs (optionally merged with macro/exogenous features)
- optionally generate per-symbol chart artifacts (HTML + indicator CSVs) into `<out-dir>/charts/`
- optionally select “best” charts meeting conditions (Ichimoku/RSI/news/Fib)
- optionally run **spectral clustering** and **DMD** across aligned multi-asset returns
- optionally output **cluster-aware portfolio target weights**

Dataset-only run:

```bash
python main.py --project hub --universe sp500 --max-symbols 25 --period 1y --interval 1d --out-dir hub_sp500_1y_1d --skip-train
```

Add chart artifacts:

```bash
python main.py --project hub --universe nasdaq100 --max-symbols 25 --period 1y --interval 1d --out-dir hub_nasdaq100_1y_1d --generate-charts --skip-train
```

Chart selection (≥ N of 4 conditions) and copies into `<out-dir>/selected_charts/`:

```bash
python main.py --project hub --universe sp500 --max-symbols 50 --period 1y --interval 1d --out-dir hub_sp500_select \
	--generate-charts --select-charts --min-conditions 3 --max-selected 50 --skip-train
```

Hub outputs:

- per-symbol datasets: `SYMBOL_period_interval.csv`
- `summary.json` (run manifest)
- optional: `macro_features.csv`, `extra_features_raw.csv`
- optional: `<out-dir>/charts/` (HTML+CSV+`charts/summary.json`)
- optional: `selected_charts.json` + `selected_charts/` copies

## Advanced analysis tools

### Fourier dominant cycles (FFT)

What it does:

- computes dominant cycle periods (in *candles*) from recent history via FFT
- adds a rolling “dominant period” feature (`FFTPeriod`) plus top-k cycle metadata columns

Where it’s useful:

- range-ish / mean-reverting markets (“rhythm”)
- exploratory diagnostics

Where it’s weaker:

- strong trends, volatility regime breaks, sparse data

### Spectral clustering (Fiedler vector)

What it does:

- builds an asset similarity graph from **aligned log-returns**
- computes the graph Laplacian and the **Fiedler vector** (2nd-smallest eigenvector)
- produces a simple **2-way clustering** (bipartition)

Why you care:

- diversification awareness (avoid many near-duplicates)
- grouping assets that move together

Example:

```bash
python main.py --project hub --symbols "ETH-USD,BTC-USD,EURUSD=X" --period 6mo --interval 1d --out-dir hub_spectral --skip-train --spectral --spectral-k 2
```

Artifacts:

- `spectral_clusters.csv`
- `spectral_graph.json`

### Dynamic Mode Decomposition (DMD)

What it does:

- decomposes the aligned multi-asset return dynamics into modes
- outputs eigenvalues with growth/decay and oscillation frequency

Why you care:

- “system-level” structure across a basket
- exploratory diagnostics of dominant behaviors

Example:

```bash
python main.py --project hub --symbols "ETH-USD,BTC-USD,EURUSD=X" --period 6mo --interval 1d --out-dir hub_dmd --skip-train --dmd --dmd-rank 3
```

Artifacts:

- `dmd_eigs.csv`
- `dmd_summary.json`

### Portfolio risk allocation across clusters (analysis-only)

This repo does **not** place trades, but it can output **target portfolio weights** to hand off to an execution layer.

If spectral clustering ran successfully, allocation can be **cluster-aware**:

- allocate weight across clusters (default: inverse-vol, option: equal)
- allocate within each cluster (default: inverse-vol, option: equal)

Allocate over the entire fetched universe:

```bash
python main.py --project hub --symbols "ETH-USD,BTC-USD,EURUSD=X" --period 6mo --interval 1d --out-dir hub_alloc \
	--skip-train --spectral --spectral-k 2 --allocate-portfolio --alloc-lookback 60
```

Allocate only over your selected charts (requires `--select-charts`):

```bash
python main.py --project hub --universe sp500 --max-symbols 50 --period 1y --interval 1d --out-dir hub_sp500_selected_alloc \
	--generate-charts --select-charts --min-conditions 3 --skip-train \
	--spectral --allocate-portfolio --alloc-universe selected
```

Artifacts:

- `portfolio_weights.csv`
- `portfolio_allocation.json`

### Systemic-risk monitoring (tensor + networks + ARIMA)

This adds a practical version of:

- Build a **time × asset × feature** tensor (defaults: `returns`, `rv` (realized vol), and optional `depth`) and run **CP/Tucker** decomposition to uncover latent factors.
- Embed assets in **2D** using **t-SNE** or **Laplacian Eigenmaps** (spectral embedding) to visualize regimes/clusters.
- Build rolling asset–asset correlation **networks**, compute **centrality/PageRank**, and fit **ARIMA** on an aggregate stress index to forecast stress build-up.

Run it standalone on an existing hub output directory:

```bash
python main.py --project systemic --in-dir hub_sp500_1y_1d
```

Or run it as part of the hub pipeline:

```bash
python main.py --project hub --universe sp500 --max-symbols 50 --period 1y --interval 1d \
	--out-dir hub_sp500_1y_1d --skip-train \
	--assets-subdir assets --shard-assets \
	--systemic-risk --tensor-method cp --tensor-rank 4 --embed tsne \
	--corr-window 60 --corr-k 8 --centrality pagerank --arima-steps 5
```

Artifacts (under `<hub_out_dir>/systemic_risk/` by default):

- `tensor_summary.json`
- `tensor_time_factors.csv`, `tensor_asset_factors.csv`
- `asset_embedding.csv` (+ `asset_embedding.html` when Plotly works)
- `centrality_timeseries.csv`
- `systemic_index.csv`
- `systemic_index_arima_forecast.json`

Note on **order-book depth**:

- If your datasets include a column containing `depth` (or you pass `--depth-col <name>`), it will be used.
- If not present, the pipeline will still run (depth becomes missing and is filled safely during tensor building).

## From analysis to execution: what a broker needs to place a trade

This repo intentionally stops at **decision + intent**. To actually trade, your broker (or broker API) needs a well-formed **order ticket**.

### The minimum “order ticket” fields

These are the core pieces of information you must decide and communicate:

1) **Instrument identification**

- `asset_class`: stock | option | future | forex | crypto
- `symbol` and/or broker-specific `instrument_id`
- `exchange/venue` (when required)
- `currency`

Extra required identifiers by asset class:

- **Options**: underlying, expiration, strike, call/put, multiplier, OCC/OPRA symbol
- **Futures**: root, month/year, contract code, multiplier, tick size, session
- **FX**: base/quote pair, settlement conventions

2) **Side + size**

- `side`: buy | sell (or sell_short depending on broker)
- `quantity` or `notional`
- position sizing rules (risk per trade %, max exposure per cluster/sector, leverage cap)

3) **Order type + prices**

- `order_type`: market | limit | stop | stop_limit | trailing_stop
- price fields by type:
	- limit: `limit_price`
	- stop: `stop_price`
	- stop-limit: both
- `time_in_force`: day | gtc | ioc | fok (plus “extended hours” flags if supported)

4) **Risk management / exits**

- stop-loss level(s)
- take-profit level(s)
- bracket/OCO settings (if supported)
- max slippage tolerance (conceptual; some brokers expose it via limit orders)

5) **Operational constraints**

- trading session (regular vs extended)
- account id / subaccount
- margin type / leverage (cash vs margin; isolated vs cross for crypto)
- compliance constraints (short locate, min tick, min order size)

### “Good practice” metadata (helps audit and debugging)

- `strategy_id` / `model_version`
- `signal_timestamp` and `data_timestamp`
- `reason`: which conditions fired (Ichimoku/RSI/news/fib), which cluster, etc.
- `confidence` / score and the inputs used

### A simple handoff format (trade intent)

If you want a clean boundary between analysis and execution, emit an **order intent JSON** alongside artifacts. An execution script can read it and talk to a broker.

Suggested structure:

```json
{
	"as_of": "2026-01-09T15:30:00Z",
	"universe": "selected",
	"portfolio": {
		"target_weights": {
			"AAPL": 0.12,
			"MSFT": 0.10
		},
		"cash_weight": 0.78
	},
	"orders": [
		{
			"asset_class": "stock",
			"symbol": "AAPL",
			"side": "buy",
			"order_type": "limit",
			"quantity": 10,
			"limit_price": 192.50,
			"time_in_force": "day",
			"risk": {"stop_loss": 185.00, "take_profit": 205.00},
			"tags": {"strategy_id": "hub_select_v1", "cluster": 0}
		}
	]
}
```

This repo does not currently write this file automatically, but the artifacts it produces (`selected_charts.json`, `portfolio_weights.csv`, etc.) contain most of the ingredients.

## Troubleshooting

### “No price data found” / Yahoo blocks

Yahoo/yfinance can be flaky (rate limits / geo blocks). The code uses fallbacks where possible, but you may still see warnings.

If data is sparse:

- try a different `--period` (e.g., `1y`) or `--interval`
- reduce the universe size (`--max-symbols`)

### Spectral/DMD/Allocation skipped

These require enough overlapping rows across assets:

- spectral/DMD: at least **3 assets** and at least `--spectral-min-overlap` aligned return rows
- allocation: at least **2 assets** and at least `--alloc-min-overlap` aligned return rows

### ML training errors

If ML complains about too few rows, use a longer history (`--period 1y`) or a faster interval.

## Disclaimer

This repository is for educational and research purposes only. It is not financial advice. Trading involves risk, including total loss of capital.
