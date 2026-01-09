import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Finance scrapers portfolio runner')
    parser.add_argument(
        '--project',
        default='collector',
        help='Which project to run: collector | sentiment_heatmap | market | ml | all',
    )
    parser.add_argument('--tickers', default=None, help='Comma-separated tickers (overrides per-project defaults)')
    parser.add_argument('--headless', action='store_true', help='Run browser automation headless (sentiment project)')
    parser.add_argument('--source', default=None, help='Sentiment headline source: rss | yahoo_static | selenium')
    parser.add_argument('--browser', default=None, help='Selenium browser for sentiment: edge | chrome')

    # Market analyzer args
    parser.add_argument('--symbols', default=None, help='Comma-separated symbols for market analyzer (e.g. BTC-USD,EURUSD=X)')
    parser.add_argument('--period', default=None, help='Market analyzer period (e.g. 6mo, 1y)')
    parser.add_argument('--interval', default=None, help='Market analyzer interval (e.g. 1d, 1h)')
    parser.add_argument('--out-dir', default=None, help='Market analyzer output directory')
    parser.add_argument('--forecast-steps', type=int, default=None, help='Market analyzer ARIMA forecast horizon')

    # ML training args (projects/ml_train.py)
    parser.add_argument('--in-csv', default=None, help='ML: input CSV path (from market analyzer)')
    parser.add_argument('--in-dir', default=None, help='ML: input directory of analyzer CSVs')
    parser.add_argument('--model', default=None, help='ML: model type (rf | hgb | xgb)')
    parser.add_argument('--task', default=None, help='ML: task (classification | regression)')
    parser.add_argument('--horizon', type=int, default=None, help='ML: label horizon in candles')
    parser.add_argument('--threshold', type=float, default=None, help='ML: classification threshold on future return')
    parser.add_argument('--test-size', type=float, default=None, help='ML: holdout fraction (time-based)')
    parser.add_argument('--cv', default=None, help='ML: validation mode (holdout | walkforward)')
    parser.add_argument('--n-splits', type=int, default=None, help='ML: walkforward folds')
    parser.add_argument('--test-window', type=int, default=None, help='ML: walkforward test window (candles)')
    parser.add_argument('--purge', type=int, default=None, help='ML: walkforward purge size (default: horizon)')
    parser.add_argument('--importance', default=None, help='ML: feature importance method (model | permutation)')
    parser.add_argument('--top-features', type=int, default=None, help='ML: top features in importance plots')
    parser.add_argument('--random-state', type=int, default=None, help='ML: random seed')
    parser.add_argument('--multi-asset', action='store_true', help='ML: combine all CSVs in --in-dir into one training set')
    args = parser.parse_args(argv)

    project = args.project.strip().lower()

    if project in {'collector', 'collect', 'multi_source', 'multi-source'}:
        from projects.main_collector import main as collector_main

        collector_args: list[str] = []
        if args.tickers:
            collector_args += ['--tickers', args.tickers]
        return collector_main(collector_args)

    if project in {'sentiment', 'sentiment_heatmap', 'heatmap'}:
        from projects.main_sentiment import main as sentiment_main

        sentiment_args: list[str] = []
        if args.tickers:
            sentiment_args += ['--tickers', args.tickers]
        if args.headless:
            sentiment_args += ['--headless']
        if args.source:
            sentiment_args += ['--source', args.source]
        if args.browser:
            sentiment_args += ['--browser', args.browser]
        return sentiment_main(sentiment_args)

    if project in {'market', 'analyzer', 'market_analyzer', 'charts'}:
        from projects.market_analyzer import main as market_main

        market_args: list[str] = []
        if args.symbols:
            market_args += ['--symbols', args.symbols]
        elif args.tickers:
            market_args += ['--symbols', args.tickers]
        if args.period:
            market_args += ['--period', args.period]
        if args.interval:
            market_args += ['--interval', args.interval]
        if args.out_dir:
            market_args += ['--out-dir', args.out_dir]
        if args.forecast_steps is not None:
            market_args += ['--forecast-steps', str(args.forecast_steps)]
        return market_main(market_args)

    if project in {'ml', 'train', 'ml_train', 'model'}:
        from projects.ml_train import main as ml_main

        ml_args: list[str] = []
        if args.in_csv:
            ml_args += ['--in-csv', args.in_csv]
        if args.in_dir:
            ml_args += ['--in-dir', args.in_dir]
        if args.model:
            ml_args += ['--model', args.model]
        if args.task:
            ml_args += ['--task', args.task]
        if args.horizon is not None:
            ml_args += ['--horizon', str(args.horizon)]
        if args.threshold is not None:
            ml_args += ['--threshold', str(args.threshold)]
        if args.test_size is not None:
            ml_args += ['--test-size', str(args.test_size)]
        if args.cv:
            ml_args += ['--cv', args.cv]
        if args.n_splits is not None:
            ml_args += ['--n-splits', str(args.n_splits)]
        if args.test_window is not None:
            ml_args += ['--test-window', str(args.test_window)]
        if args.purge is not None:
            ml_args += ['--purge', str(args.purge)]
        if args.importance:
            ml_args += ['--importance', args.importance]
        if args.top_features is not None:
            ml_args += ['--top-features', str(args.top_features)]
        if args.random_state is not None:
            ml_args += ['--random-state', str(args.random_state)]
        if args.multi_asset:
            ml_args += ['--multi-asset']
        if args.out_dir:
            ml_args += ['--out-dir', args.out_dir]
        return ml_main(ml_args)

    if project == 'all':
        from projects.main_collector import main as collector_main
        from projects.main_sentiment import main as sentiment_main

        collector_args: list[str] = []
        sentiment_args: list[str] = []
        if args.tickers:
            collector_args += ['--tickers', args.tickers]
            sentiment_args += ['--tickers', args.tickers]
        if args.headless:
            sentiment_args += ['--headless']
        if args.source:
            sentiment_args += ['--source', args.source]
        if args.browser:
            sentiment_args += ['--browser', args.browser]

        rc1 = collector_main(collector_args)
        rc2 = sentiment_main(sentiment_args)
        return rc1 or rc2

    raise SystemExit(f"Unknown --project '{args.project}'. Try: collector | sentiment_heatmap | market | ml | all")


if __name__ == '__main__':
    raise SystemExit(main())
