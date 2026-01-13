"""Systemic-risk / market microstructure monitoring utilities.

This module intentionally builds on the repo's existing data hub outputs:
per-symbol OHLCV CSVs written by `projects/data_hub_train.py`.

Key ideas implemented here (lightweight, practical):

1) Tensor structure
   - Build a time × asset × feature tensor from aligned series.
   - Features include: log returns, realized volatility, and optional order-book depth.
   - Run CP or Tucker decomposition (via tensorly) to uncover latent factors.

2) Regime visualization
   - Reduce asset factor loadings to 2D via t-SNE or Laplacian Eigenmaps
     (sklearn SpectralEmbedding) to visualize clusters/regimes.

3) Dynamic correlation networks
   - Build rolling correlation graphs across assets.
   - Compute centrality / PageRank per window to identify systemically important assets.
   - Fit ARIMA to a systemic index to forecast stress build-up.

Research/education only; not financial advice.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TensorBuildResult:
    dates: pd.DatetimeIndex
    assets: list[str]
    feature_names: list[str]
    tensor: np.ndarray  # (T, A, F)


def _is_dataset_csv(path: Path) -> bool:
    n = path.name.lower()
    if not n.endswith('.csv'):
        return False
    if n.endswith('_importance.csv'):
        return False
    if n in {'macro_features.csv', 'extra_features_raw.csv'}:
        return False
    return True


def _list_csvs_shallow(root: Path) -> list[Path]:
    """List CSVs in `root` plus one sharding level.

    Avoid deep recursion: hub output trees can be massive on Windows.
    """
    out: list[Path] = []
    out.extend([p for p in root.glob('*.csv') if _is_dataset_csv(p)])
    try:
        for child in root.iterdir():
            if child.is_dir():
                out.extend([p for p in child.glob('*.csv') if _is_dataset_csv(p)])
    except FileNotFoundError:
        pass
    return out


def discover_hub_datasets(
    hub_dir: str | Path,
    *,
    assets_subdir: str | None = None,
) -> list[Path]:
    """Discover per-symbol dataset CSVs written by the hub.

    Supports:
    - legacy flat layout: <hub_dir>/*.csv
    - assets layout: <hub_dir>/<assets_subdir>/*.csv
    - sharded assets: <hub_dir>/<assets_subdir>/<prefix>/*.csv
    """
    d = Path(hub_dir)
    if not d.exists():
        raise FileNotFoundError(str(d))

    candidates: list[Path] = []

    # Prefer user-specified assets subdir, otherwise the conventional 'assets'.
    if assets_subdir:
        ad = d / str(assets_subdir)
        if ad.exists() and ad.is_dir():
            candidates = _list_csvs_shallow(ad)

    if not candidates:
        ad = d / 'assets'
        if ad.exists() and ad.is_dir():
            candidates = _list_csvs_shallow(ad)

    if not candidates:
        candidates = _list_csvs_shallow(d)

    return sorted(candidates)


def _load_csv_time_indexed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    date_col = None
    for c in ('Date', 'Datetime', 'date', 'datetime'):
        if c in df.columns:
            date_col = c
            break
    if date_col is None and df.columns.size and str(df.columns[0]).lower().startswith('unnamed'):
        date_col = df.columns[0]

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    return df


def _infer_symbol_from_filename(path: Path) -> str:
    # Expected: SYMBOL_period_interval.csv
    stem = path.stem
    if '_' in stem:
        return stem.split('_', 1)[0].strip().upper()
    return stem.strip().upper()


def _safe_log_returns(close: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors='coerce').astype(float)
    c = c.replace([np.inf, -np.inf], np.nan)
    return np.log(c).diff()


def _realized_vol(returns: pd.Series, *, window: int = 20) -> pd.Series:
    r = pd.to_numeric(returns, errors='coerce').astype(float)
    return r.rolling(int(window), min_periods=max(5, int(window) // 2)).std()


def _find_depth_column(df: pd.DataFrame, depth_col: str | None) -> str | None:
    if df is None or df.empty:
        return None
    if depth_col:
        if depth_col in df.columns:
            return depth_col
        # case-insensitive match
        for c in df.columns:
            if str(c).strip().lower() == str(depth_col).strip().lower():
                return str(c)
        return None

    # heuristic: any column containing 'depth'
    for c in df.columns:
        if 'depth' in str(c).lower():
            return str(c)
    return None


def build_time_asset_feature_tensor(
    dataset_paths: list[Path],
    *,
    features: list[str] | None = None,
    vol_window: int = 20,
    depth_col: str | None = None,
    align: str = 'inner',
) -> TensorBuildResult:
    """Build a (T, A, F) tensor from hub dataset CSVs.

    features options (strings):
    - returns
    - rv (realized volatility)
    - depth (order-book depth if present, else NaN)
    - dollar_volume (Close * Volume)
    - hl_range (High-Low normalized by Close)

    `align`:
    - inner: intersection of all timestamps
    - outer: union of timestamps (missing filled)

    Notes:
    - Tensor decompositions require numeric values; we z-score per feature and fill NaNs with 0.
    """
    feats = [f.strip().lower() for f in (features or ['returns', 'rv', 'depth']) if str(f).strip()]
    if not feats:
        feats = ['returns', 'rv', 'depth']

    frames: dict[str, pd.DataFrame] = {}
    for p in dataset_paths:
        sym = _infer_symbol_from_filename(p)
        df = _load_csv_time_indexed(p)
        if df.empty:
            continue
        frames[sym] = df

    assets = sorted(frames.keys())
    if len(assets) < 2:
        raise ValueError('Need at least 2 assets to build a multi-asset tensor.')

    # Build feature panels (one DataFrame per feature: index=time, columns=asset)
    panels: dict[str, pd.Series] = {}

    # First compute per-asset series, then align.
    series_by_feature: dict[str, dict[str, pd.Series]] = {f: {} for f in feats}

    for sym in assets:
        df = frames[sym]
        close = df.get('Close')
        if close is None:
            continue
        rets = _safe_log_returns(close)

        if 'returns' in feats:
            series_by_feature['returns'][sym] = rets
        if 'rv' in feats:
            series_by_feature['rv'][sym] = _realized_vol(rets, window=vol_window)
        if 'dollar_volume' in feats:
            vol = pd.to_numeric(df.get('Volume'), errors='coerce') if 'Volume' in df.columns else None
            if vol is None:
                series_by_feature['dollar_volume'][sym] = pd.Series(index=rets.index, data=np.nan)
            else:
                series_by_feature['dollar_volume'][sym] = pd.to_numeric(close, errors='coerce') * vol
        if 'hl_range' in feats:
            if 'High' in df.columns and 'Low' in df.columns:
                hi = pd.to_numeric(df['High'], errors='coerce')
                lo = pd.to_numeric(df['Low'], errors='coerce')
                c = pd.to_numeric(close, errors='coerce')
                series_by_feature['hl_range'][sym] = (hi - lo) / c.replace(0.0, np.nan)
            else:
                series_by_feature['hl_range'][sym] = pd.Series(index=rets.index, data=np.nan)
        if 'depth' in feats:
            dc = _find_depth_column(df, depth_col)
            if dc is not None:
                series_by_feature['depth'][sym] = pd.to_numeric(df[dc], errors='coerce')
            else:
                series_by_feature['depth'][sym] = pd.Series(index=rets.index, data=np.nan)

    # Align on time
    panel_dfs: list[pd.DataFrame] = []
    feature_names: list[str] = []

    how = (align or 'inner').strip().lower()
    if how not in {'inner', 'outer'}:
        how = 'inner'

    for f in feats:
        per_asset = series_by_feature.get(f) or {}
        if len(per_asset) < 2:
            continue
        dfp = pd.DataFrame(per_asset)
        # remove columns with all NaNs
        dfp = dfp.dropna(axis=1, how='all')
        if dfp.shape[1] < 2:
            continue
        panel_dfs.append(dfp)
        feature_names.append(f)

    if not panel_dfs:
        raise ValueError('No usable features found to build the tensor.')

    # Build a shared time index and asset set.
    # Start from the first panel and align others onto it.
    base = panel_dfs[0]
    for dfp in panel_dfs[1:]:
        base, dfp2 = base.align(dfp, join=how, axis=0)
        # Keep base updated; other frames will be re-aligned later.
        base = base

    dates = base.index

    # Determine common assets across panels (intersection, to keep a rectangular tensor).
    assets_final = sorted(set(base.columns))
    for dfp in panel_dfs[1:]:
        assets_final = sorted(set(assets_final).intersection(set(dfp.columns)))

    if len(assets_final) < 2:
        raise ValueError('After alignment, fewer than 2 assets remain with data.')

    # Rebuild a tensor with consistent (dates, assets)
    T = len(dates)
    A = len(assets_final)
    F = len(feature_names)

    tensor = np.zeros((T, A, F), dtype=float)

    for j, f in enumerate(feature_names):
        dfp = pd.DataFrame(series_by_feature[f])
        dfp = dfp.reindex(index=dates, columns=assets_final)

        # Z-score feature globally (across time×asset) to make decompositions stable.
        vals = dfp.to_numpy(dtype=float)
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        z = (vals - mu) / sd
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        tensor[:, :, j] = z

    return TensorBuildResult(dates=dates, assets=assets_final, feature_names=feature_names, tensor=tensor)


def cp_decompose(tensor: np.ndarray, *, rank: int = 4, random_state: int = 42):
    try:
        import tensorly as tl  # type: ignore[import-not-found]
        from tensorly.decomposition import parafac  # type: ignore[import-not-found]

        tl.set_backend('numpy')
        # parafac returns a CPTensor (weights, factors)
        return parafac(tensor, rank=int(rank), random_state=int(random_state), init='svd', tol=1e-6)
    except ImportError as e:
        raise ImportError('tensorly is required for CP decomposition. Install it via requirements.txt.') from e


def tucker_decompose(tensor: np.ndarray, *, ranks: tuple[int, int, int] = (4, 4, 3), random_state: int = 42):
    try:
        import tensorly as tl  # type: ignore[import-not-found]
        from tensorly.decomposition import tucker  # type: ignore[import-not-found]

        tl.set_backend('numpy')
        return tucker(tensor, rank=tuple(int(x) for x in ranks), random_state=int(random_state), init='svd', tol=1e-6)
    except ImportError as e:
        raise ImportError('tensorly is required for Tucker decomposition. Install it via requirements.txt.') from e


def embed_assets_2d(asset_factors: np.ndarray, *, method: str = 'tsne', random_state: int = 42) -> np.ndarray:
    X = np.asarray(asset_factors, dtype=float)
    m = (method or 'tsne').strip().lower()

    if X.shape[0] < 3:
        # Too few points to embed.
        return np.zeros((X.shape[0], 2), dtype=float)

    if m in {'laplacian', 'eigenmaps', 'spectral', 'laplacian_eigenmaps'}:
        from sklearn.manifold import SpectralEmbedding

        emb = SpectralEmbedding(n_components=2, random_state=int(random_state), affinity='nearest_neighbors')
        return np.asarray(emb.fit_transform(X), dtype=float)

    # default: TSNE
    from sklearn.manifold import TSNE

    # Perplexity must be < n_samples
    perplexity = min(30.0, max(2.0, float((X.shape[0] - 1) // 3)))
    tsne = TSNE(n_components=2, random_state=int(random_state), perplexity=perplexity, init='pca', learning_rate='auto')
    return np.asarray(tsne.fit_transform(X), dtype=float)


def rolling_correlation_networks(
    returns: pd.DataFrame,
    *,
    window: int = 60,
    k: int = 8,
    use_abs: bool = True,
) -> list[pd.DataFrame]:
    """Build rolling kNN correlation adjacency matrices.

    Returns a list of adjacency DataFrames aligned to end-of-window timestamps.
    """
    if returns is None or returns.empty:
        return []

    from utils.spectral import similarity_from_returns, knn_adjacency

    window = int(window)
    k = int(k)

    out: list[pd.DataFrame] = []
    for end in range(window, returns.shape[0] + 1):
        sl = returns.iloc[end - window : end]
        sim = similarity_from_returns(sl, method='pearson', use_abs=bool(use_abs))
        if sim.empty:
            continue
        adj = knn_adjacency(sim, k=k, symmetric=True)
        adj.attrs['timestamp'] = returns.index[end - 1]
        out.append(adj)

    return out


def centrality_time_series(
    adjs: list[pd.DataFrame],
    *,
    method: str = 'pagerank',
) -> pd.DataFrame:
    """Compute a centrality score per asset per window."""
    if not adjs:
        return pd.DataFrame()

    try:
        import networkx as nx  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError('networkx is required for centrality/PageRank. Install it via requirements.txt.') from e

    m = (method or 'pagerank').strip().lower()

    rows: list[pd.Series] = []
    idx: list[pd.Timestamp] = []

    for adj in adjs:
        ts = adj.attrs.get('timestamp')
        ts = pd.to_datetime(ts) if ts is not None else None

        G = nx.from_pandas_adjacency(adj.astype(float), create_using=nx.Graph)

        if m in {'eig', 'eigen', 'eigenvector'}:
            # eigenvector centrality can fail on disconnected graphs; catch and fall back.
            try:
                scores = nx.eigenvector_centrality_numpy(G, weight='weight')
            except Exception:
                scores = nx.pagerank(G, weight='weight')
        elif m in {'between', 'betweenness'}:
            scores = nx.betweenness_centrality(G, weight='weight', normalized=True)
        else:
            scores = nx.pagerank(G, weight='weight')

        s = pd.Series(scores, dtype=float)
        # Ensure consistent column ordering
        s = s.reindex(adj.index).astype(float)
        rows.append(s)
        idx.append(ts if ts is not None else pd.Timestamp('1970-01-01'))

    dfc = pd.DataFrame(rows, index=pd.DatetimeIndex(idx)).sort_index()
    dfc.index.name = 'Date'
    return dfc


def _fit_arima_forecast(series: pd.Series, *, steps: int = 5, order: tuple[int, int, int] = (1, 0, 1)) -> dict:
    y = pd.to_numeric(series, errors='coerce').dropna().astype(float)
    if y.size < 60:
        last = float(y.iloc[-1]) if y.size else float('nan')
        return {
            'order': list(order),
            'steps': int(steps),
            'forecast': [last] * int(steps),
            'ok': False,
            'note': 'Too few points for ARIMA; using naive forecast.',
        }

    try:
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = ARIMA(y, order=order)
            fitted = model.fit()
            fc = fitted.forecast(steps=int(steps))

        return {
            'order': list(order),
            'steps': int(steps),
            'forecast': [float(v) for v in fc.values],
            'ok': True,
        }
    except Exception as e:
        last = float(y.iloc[-1])
        return {
            'order': list(order),
            'steps': int(steps),
            'forecast': [last] * int(steps),
            'ok': False,
            'note': f'ARIMA failed ({e}); using naive forecast.',
        }


def run_systemic_risk(
    *,
    hub_dir: str | Path,
    out_dir: str | Path | None = None,
    assets_subdir: str | None = None,
    tensor_method: str = 'cp',
    tensor_rank: int = 4,
    tucker_ranks: tuple[int, int, int] = (4, 4, 3),
    embed_method: str = 'tsne',
    features: list[str] | None = None,
    vol_window: int = 20,
    depth_col: str | None = None,
    corr_window: int = 60,
    corr_k: int = 8,
    centrality: str = 'pagerank',
    arima_steps: int = 5,
    random_state: int = 42,
) -> dict:
    """Run the full systemic-risk pipeline and write artifacts."""

    hub_dir = Path(hub_dir)
    out_base = Path(out_dir) if out_dir is not None else (hub_dir / 'systemic_risk')
    out_base.mkdir(parents=True, exist_ok=True)

    dataset_paths = discover_hub_datasets(hub_dir, assets_subdir=assets_subdir)
    if not dataset_paths:
        raise ValueError(f'No dataset CSVs found under: {hub_dir}')

    t = build_time_asset_feature_tensor(
        dataset_paths,
        features=features,
        vol_window=int(vol_window),
        depth_col=depth_col,
        align='inner',
    )

    # --- Tensor decomposition ---
    tensor_summary: dict = {
        'assets': t.assets,
        'features': t.feature_names,
        'tensor_shape': list(t.tensor.shape),
        'method': str(tensor_method),
    }

    asset_factors = None
    time_factors = None

    m = (tensor_method or 'cp').strip().lower()
    if m in {'tucker'}:
        core, factors = tucker_decompose(t.tensor, ranks=tucker_ranks, random_state=random_state)
        # factors is [time_factor, asset_factor, feature_factor]
        time_factors = np.asarray(factors[0], dtype=float)
        asset_factors = np.asarray(factors[1], dtype=float)
        tensor_summary.update({'tucker_ranks': [int(x) for x in tucker_ranks]})
    else:
        cp = cp_decompose(t.tensor, rank=int(tensor_rank), random_state=random_state)
        # cp.factors order matches tensor modes: time, asset, feature
        factors = cp.factors
        time_factors = np.asarray(factors[0], dtype=float)
        asset_factors = np.asarray(factors[1], dtype=float)
        tensor_summary.update({'cp_rank': int(tensor_rank)})

    # Save time factors
    tf = pd.DataFrame(time_factors, index=t.dates)
    tf.index.name = 'Date'
    tf.to_csv(out_base / 'tensor_time_factors.csv')

    af = pd.DataFrame(asset_factors, index=t.assets)
    af.index.name = 'symbol'
    af.to_csv(out_base / 'tensor_asset_factors.csv')

    with open(out_base / 'tensor_summary.json', 'w', encoding='utf-8') as f:
        json.dump(tensor_summary, f, indent=2)

    # --- 2D embedding of assets (regime visualization / clustering) ---
    coords = embed_assets_2d(asset_factors, method=embed_method, random_state=random_state)
    df_emb = pd.DataFrame({'symbol': t.assets, 'x': coords[:, 0], 'y': coords[:, 1]})
    df_emb.to_csv(out_base / 'asset_embedding.csv', index=False)

    try:
        import plotly.express as px

        fig = px.scatter(df_emb, x='x', y='y', text='symbol', hover_name='symbol', title='Asset embedding (2D)')
        fig.update_traces(textposition='top center')
        fig.update_layout(template='plotly_white')
        fig.write_html(out_base / 'asset_embedding.html')
    except Exception:
        # Plotly is optional for this artifact.
        pass

    # --- Dynamic correlation network + centrality time series ---
    # Build aligned returns panel (reuse tensor's dates/assets and recompute returns directly from CSVs)
    closes: dict[str, pd.Series] = {}
    for p in dataset_paths:
        sym = _infer_symbol_from_filename(p)
        if sym not in set(t.assets):
            continue
        df = _load_csv_time_indexed(p)
        if 'Close' not in df.columns:
            continue
        closes[sym] = pd.to_numeric(df['Close'], errors='coerce')

    close_panel = pd.DataFrame({s: closes[s] for s in t.assets}).reindex(index=t.dates)
    rets = np.log(close_panel).diff().replace([np.inf, -np.inf], np.nan)

    adjs = rolling_correlation_networks(rets.dropna(how='all'), window=int(corr_window), k=int(corr_k), use_abs=True)
    cent = centrality_time_series(adjs, method=centrality)
    cent.to_csv(out_base / 'centrality_timeseries.csv')

    # Systemic index: mean of top-5 centrality each window (simple early-warning aggregate)
    systemic = cent.apply(lambda r: float(np.nanmean(np.sort(r.values)[-min(5, r.size):])) if r.size else np.nan, axis=1)
    systemic.name = 'systemic_index_top5_mean'
    systemic.to_frame().to_csv(out_base / 'systemic_index.csv')

    fc = _fit_arima_forecast(systemic, steps=int(arima_steps), order=(1, 0, 1))
    with open(out_base / 'systemic_index_arima_forecast.json', 'w', encoding='utf-8') as f:
        json.dump(fc, f, indent=2)

    return {
        'out_dir': str(out_base),
        'artifacts': {
            'tensor_summary_json': str(out_base / 'tensor_summary.json'),
            'tensor_time_factors_csv': str(out_base / 'tensor_time_factors.csv'),
            'tensor_asset_factors_csv': str(out_base / 'tensor_asset_factors.csv'),
            'asset_embedding_csv': str(out_base / 'asset_embedding.csv'),
            'asset_embedding_html': str(out_base / 'asset_embedding.html'),
            'centrality_timeseries_csv': str(out_base / 'centrality_timeseries.csv'),
            'systemic_index_csv': str(out_base / 'systemic_index.csv'),
            'systemic_index_arima_forecast_json': str(out_base / 'systemic_index_arima_forecast.json'),
        },
        'params': {
            'tensor_method': str(tensor_method),
            'tensor_rank': int(tensor_rank),
            'tucker_ranks': [int(x) for x in tucker_ranks],
            'embed_method': str(embed_method),
            'features': features or ['returns', 'rv', 'depth'],
            'vol_window': int(vol_window),
            'depth_col': depth_col,
            'corr_window': int(corr_window),
            'corr_k': int(corr_k),
            'centrality': str(centrality),
            'arima_steps': int(arima_steps),
        },
    }
