"""Systemic-risk monitoring CLI.

Runs tensor + network systemic-risk analysis on hub outputs.

Examples:
  python main.py --project systemic --in-dir hub_sp500_1y_1d

Or run it after the hub in one shot:
  python main.py --project hub ... --systemic-risk

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import os
import sys


# Allow running as a script (python projects\systemic_risk.py) without import errors.
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.systemic_risk import run_systemic_risk


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Systemic-risk monitoring (tensor + networks + ARIMA)')

    p.add_argument('--in-dir', required=True, help='Hub output directory (contains per-symbol CSVs)')
    p.add_argument('--out-dir', default=None, help='Where to write systemic-risk artifacts (default: <in-dir>/systemic_risk)')

    # layout
    p.add_argument('--assets-subdir', default=None, help="If datasets are under a subfolder (e.g. 'assets'), specify it")

    # tensor
    p.add_argument('--tensor-method', default='cp', help='cp | tucker')
    p.add_argument('--tensor-rank', type=int, default=4, help='CP rank')
    p.add_argument('--tucker-ranks', default='4,4,3', help='Tucker ranks as i,j,k (time,asset,feature)')
    p.add_argument('--features', default='returns,rv,depth', help='Comma-separated tensor features')
    p.add_argument('--vol-window', type=int, default=20, help='Realized volatility lookback window')
    p.add_argument('--depth-col', default=None, help='Optional column name for order-book depth')

    # embedding
    p.add_argument('--embed', default='tsne', help='tsne | laplacian')

    # network + ARIMA
    p.add_argument('--corr-window', type=int, default=60, help='Rolling correlation window size')
    p.add_argument('--corr-k', type=int, default=8, help='kNN edges per node')
    p.add_argument('--centrality', default='pagerank', help='pagerank | eigenvector | betweenness')
    p.add_argument('--arima-steps', type=int, default=5, help='ARIMA forecast horizon on systemic index')

    p.add_argument('--random-state', type=int, default=42)

    args = p.parse_args(argv)

    tucker_ranks = tuple(int(x.strip()) for x in str(args.tucker_ranks).split(',') if x.strip())
    if len(tucker_ranks) != 3:
        raise SystemExit('--tucker-ranks must be 3 integers like 4,4,3')

    features = [x.strip() for x in str(args.features).split(',') if x.strip()]

    res = run_systemic_risk(
        hub_dir=args.in_dir,
        out_dir=args.out_dir,
        assets_subdir=args.assets_subdir,
        tensor_method=args.tensor_method,
        tensor_rank=int(args.tensor_rank),
        tucker_ranks=tucker_ranks,  # type: ignore[arg-type]
        embed_method=args.embed,
        features=features,
        vol_window=int(args.vol_window),
        depth_col=args.depth_col,
        corr_window=int(args.corr_window),
        corr_k=int(args.corr_k),
        centrality=args.centrality,
        arima_steps=int(args.arima_steps),
        random_state=int(args.random_state),
    )

    print('Wrote systemic-risk artifacts to:', res.get('out_dir'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
