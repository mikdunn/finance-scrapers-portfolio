import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Finance scrapers portfolio runner', allow_abbrev=False)
    parser.add_argument(
        '--project',
        default='collector',
        help='Which project to run: collector | sentiment_heatmap | market | ml | hub | backtest | systemic | sweep | report | all',
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
    parser.add_argument(
        '--in-dir',
        default=None,
        help='Input directory of analyzer/hub CSVs (used by ML via --project ml and by systemic via --project systemic)',
    )
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
    # Systemic risk / microstructure monitoring
    # Allow project-specific flags to pass through for projects that have their own CLI.
    args, unknown = parser.parse_known_args(argv)

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
        return ml_main(ml_args + unknown)

    if project in {'hub', 'data_hub', 'data-hub', 'dataset'}:
        from projects.data_hub_train import main as hub_main

        # Forward any remaining args to the hub CLI.
        hub_args: list[str] = []
        # If user provided tickers/symbols at top-level, map them to hub --symbols.
        if args.tickers and not any(x in unknown for x in ['--symbols', '--symbols-file', '--universe']):
            hub_args += ['--symbols', args.tickers]
        if args.symbols and not any(x in unknown for x in ['--symbols', '--symbols-file', '--universe']):
            hub_args += ['--symbols', args.symbols]
        if args.period and '--period' not in unknown:
            hub_args += ['--period', args.period]
        if args.interval and '--interval' not in unknown:
            hub_args += ['--interval', args.interval]
        if args.out_dir and '--out-dir' not in unknown:
            hub_args += ['--out-dir', args.out_dir]
        return hub_main(hub_args + unknown)

    if project in {'systemic', 'systemic_risk', 'risk', 'microstructure'}:
        from projects.systemic_risk import main as systemic_main

        sys_args: list[str] = []
        if args.in_dir:
            sys_args += ['--in-dir', args.in_dir]
        elif args.out_dir:
            # Convenience: if user passes --out-dir at top-level and it points to a hub directory,
            # treat it as the input directory.
            sys_args += ['--in-dir', args.out_dir]

        # Forward remaining args to the systemic CLI.
        return systemic_main(sys_args + unknown)

    if project in {'backtest', 'strategy', 'sim', 'simulator'}:
        from projects.strategy_backtest import main as backtest_main

        bt_args: list[str] = []
        # NOTE: main.py defines --model/--task/--threshold for the ML project.
        # When running --project backtest, argparse will still consume these.
        # Forward them explicitly so the backtest CLI receives them.
        if args.model:
            bt_args += ['--model', args.model]
        if args.task:
            bt_args += ['--task', args.task]
        if args.threshold is not None:
            bt_args += ['--threshold', str(args.threshold)]
        if args.in_csv:
            bt_args += ['--in-csv', args.in_csv]
        if args.in_dir:
            bt_args += ['--in-dir', args.in_dir]
        if args.out_dir:
            bt_args += ['--out-dir', args.out_dir]

        # Forward remaining args (including required --model)
        return backtest_main(bt_args + unknown)

    if project in {'sweep', 'strategy_sweep', 'sweeper', 'experiments'}:
        from projects.strategy_sweep import main as sweep_main

        sweep_args: list[str] = []

        if args.in_csv:
            sweep_args += ['--in-csv', args.in_csv]
        if args.horizon is not None:
            sweep_args += ['--horizon', str(args.horizon)]
        if args.threshold is not None and '--label-threshold' not in unknown:
            # Map the top-level ML flag name (--threshold) to sweep's label threshold.
            sweep_args += ['--label-threshold', str(args.threshold)]
        if args.out_dir and '--out-dir' not in unknown:
            sweep_args += ['--out-dir', str(args.out_dir)]

        return sweep_main(sweep_args + unknown)

    if project in {'report', 'backtest_report', 'compare', 'comparison'}:
        from projects.backtest_report import main as report_main

        rep_args: list[str] = []

        # main.py defines --out-dir for other projects; forward it here as a convenience
        # when the user passes it at the top level.
        if args.out_dir and '--out-dir' not in unknown:
            rep_args += ['--out-dir', args.out_dir]

        # Forward remaining args to the report CLI.
        return report_main(rep_args + unknown)

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

    raise SystemExit(
        f"Unknown --project '{args.project}'. Try: collector | sentiment_heatmap | market | ml | hub | backtest | systemic | sweep | report | all"
    )


if __name__ == '__main__':
    raise SystemExit(main())
