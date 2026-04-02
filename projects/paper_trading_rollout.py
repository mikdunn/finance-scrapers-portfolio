"""Phase 5: paper trading rollout and validation CLI."""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from projects.strategy_runner import main as strategy_runner_main
from utils.paper_trading_validation import (
    PaperTradingConfig,
    PaperTradingValidator,
    build_phase6_tuning_plan,
    build_validation_result_from_artifacts,
    calibrate_promotion_gates,
    create_paper_trading_validation_report,
    save_phase6_tuning_plan,
    save_promotion_gate_calibration,
    save_validation_result_json,
)


SMOKE_MODELS = {
    "eurusd": "artifacts/smoke/smoke_ml_eurusd_6mo/model.joblib",
    "btc": "artifacts/smoke/smoke_ml_btc_6mo/model.joblib",
    "multi": "artifacts/smoke/smoke_ml_multi_alloc_smoke/model.joblib",
}


def _resolve_smoke_model(name: str) -> Path:
    key = str(name).strip().lower()
    model_path = Path(SMOKE_MODELS.get(key, key))
    if not model_path.exists():
        raise FileNotFoundError(f"Smoke model not found: {model_path}")
    return model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or validate a Phase 5 paper trading rollout")
    parser.add_argument("--out-dir", default="artifacts/paper_trading", help="Output directory for validation artifacts")
    parser.add_argument("--metrics-file", default=None, help="Existing runner_metrics.json to validate")
    parser.add_argument("--events-file", default=None, help="Existing events.jsonl to validate")
    parser.add_argument("--run-runner", action="store_true", help="Run strategy_runner first, then validate its artifacts")
    parser.add_argument("--run-smoke", action="store_true", help="Run a fresh smoke paper-trading pass using a built-in smoke model")
    parser.add_argument("--smoke-model", default="eurusd", help="eurusd | btc | multi | explicit path to model.joblib")

    parser.add_argument("--model", default=None, help="Model path for strategy_runner when --run-runner is used")
    parser.add_argument("--symbol", default="EURUSD=X", help="Single symbol to paper trade")
    parser.add_argument("--period", default="7d", help="Historical period for strategy_runner")
    parser.add_argument("--interval", default="1m", help="Data interval for strategy_runner")
    parser.add_argument("--qty", type=float, default=0.01, help="Order quantity for runner")
    parser.add_argument("--mode", default="dry_run_delayed", help="dry_run | dry_run_delayed | alpaca")
    parser.add_argument("--execution-profile", default="scalper", help="scalper | moderate | conservative")
    parser.add_argument("--volatility-regime", type=float, default=1.0, help="Volatility multiplier")
    parser.add_argument("--max-iterations", type=int, default=5, help="Runner iterations when --run-runner is used")
    parser.add_argument("--poll-seconds", type=float, default=0.2, help="Runner loop sleep")
    parser.add_argument("--dashboard-enabled", action="store_true", help="Enable phase-4 dashboard during rollout")

    parser.add_argument("--broker", default="paper_simulator", help="alpaca | paper_simulator")
    parser.add_argument("--base-capital", type=float, default=10000.0, help="Starting capital for validation scoring")
    parser.add_argument("--min-trades-for-validation", type=int, default=1, help="Minimum closed trades to consider the run valid")
    parser.add_argument("--target-win-rate-pct", type=float, default=50.0, help="Target win rate percentage")
    parser.add_argument("--max-consecutive-loss-streak", type=int, default=3, help="Maximum allowed consecutive losses")
    parser.add_argument("--calibrate-gates", action="store_true", help="Calibrate promotion gates from existing validation reports")
    parser.add_argument("--reports-glob", default="artifacts/backtests/**/paper_trading_validation.json", help="Glob for historical validation reports")
    parser.add_argument("--min-reports-required", type=int, default=3, help="Minimum reports before hard gate enforcement")
    parser.add_argument("--start-phase6", action="store_true", help="Generate a Phase 6 tuning plan from validation reports")

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(args.metrics_file) if args.metrics_file else out_dir / "runner_metrics.json"
    events_file = Path(args.events_file) if args.events_file else out_dir / "events.jsonl"
    dashboard_file = out_dir / "dashboard.html"

    if args.run_smoke:
        smoke_model = _resolve_smoke_model(str(args.smoke_model))
        smoke_alias = str(args.smoke_model).strip().lower()
        if smoke_alias == "btc":
            args.symbol = "BTC-USD"
        elif smoke_alias == "eurusd":
            args.symbol = "EURUSD=X"
        elif smoke_alias == "multi":
            args.symbol = "BTC-USD"
        args.model = str(smoke_model)
        args.run_runner = True

    if args.calibrate_gates:
        report_files = sorted(glob(str(args.reports_glob), recursive=True))
        calibration = calibrate_promotion_gates(
            report_files,
            min_reports_required=int(args.min_reports_required),
        )
        gate_json = out_dir / "promotion_gate_calibration.json"
        save_promotion_gate_calibration(calibration, str(gate_json))
        print(f"Wrote: {gate_json}")

        if args.start_phase6:
            phase6_plan = build_phase6_tuning_plan(report_files)
            phase6_json = out_dir / "phase6_tuning_plan.json"
            save_phase6_tuning_plan(phase6_plan, str(phase6_json))
            print(f"Wrote: {phase6_json}")
        return 0

    if args.run_runner:
        if not args.model:
            raise SystemExit("--model is required when --run-runner is used")
        runner_args = [
            "--model", args.model,
            "--symbol", args.symbol,
            "--period", args.period,
            "--interval", args.interval,
            "--qty", str(args.qty),
            "--mode", args.mode,
            "--execution-profile", args.execution_profile,
            "--volatility-regime", str(args.volatility_regime),
            "--max-iterations", str(args.max_iterations),
            "--poll-seconds", str(args.poll_seconds),
            "--metrics-file", str(metrics_file),
            "--events-file", str(events_file),
            "--state-file", str(out_dir / "runtime_state.json"),
            "--dashboard-file", str(dashboard_file),
        ]
        if args.dashboard_enabled:
            runner_args.append("--dashboard-enabled")
        rc = strategy_runner_main(runner_args)
        if int(rc) != 0:
            raise SystemExit(int(rc))

    cfg = PaperTradingConfig(
        broker=str(args.broker),
        base_capital=float(args.base_capital),
        symbol=str(args.symbol),
        execution_profile=str(args.execution_profile),
        volatility_regime=float(args.volatility_regime),
        min_trades_for_validation=int(args.min_trades_for_validation),
        target_win_rate_pct=float(args.target_win_rate_pct),
        max_consecutive_loss_streak=int(args.max_consecutive_loss_streak),
        dashboard_enabled=bool(args.dashboard_enabled),
        max_iterations=int(args.max_iterations),
    )
    validator = PaperTradingValidator(cfg)
    if not validator.validate_config():
        raise SystemExit("Invalid paper trading configuration: " + "; ".join(validator.validation_errors))

    result = build_validation_result_from_artifacts(
        config=cfg,
        metrics_file=str(metrics_file),
        events_file=str(events_file),
        run_id=validator.run_id,
    )
    validator.check_validation_criteria(result)

    json_out = out_dir / "paper_trading_validation.json"
    html_out = out_dir / "paper_trading_validation.html"
    save_validation_result_json(result, str(json_out))
    create_paper_trading_validation_report(result, str(html_out))

    status = "PASSED" if result.passed_validation else "REVIEW_NEEDED"
    print(f"Paper trading validation {status}")
    print(f"Wrote: {json_out}")
    print(f"Wrote: {html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())