"""Phase 6: performance tuning and promotion heuristics CLI."""

from __future__ import annotations

import argparse
import csv
import json
from glob import glob
from pathlib import Path

from projects.paper_trading_rollout import main as paper_rollout_main
from utils.paper_trading_validation import (
    build_phase6_tuning_plan,
    calibrate_promotion_gates,
    save_phase6_tuning_plan,
    save_promotion_gate_calibration,
)


def _parse_profiles(value: str) -> list[str]:
    profiles = [p.strip().lower() for p in str(value).split(",") if p.strip()]
    allowed = {"scalper", "moderate", "conservative"}
    return [p for p in profiles if p in allowed]


def _score_candidate(row: dict) -> float:
    passed = 1.0 if bool(row.get("passed_validation")) else 0.0
    total_trades = float(row.get("total_trades", 0.0) or 0.0)
    win_rate = float(row.get("win_rate_pct", 0.0) or 0.0)
    pnl_pct = float(row.get("gross_pnl_pct", 0.0) or 0.0)
    p95_ms = float(row.get("p95_latency_ms", 0.0) or 0.0)

    # Weighted composite score for promotion ranking.
    return (
        passed * 40.0
        + min(total_trades, 50.0) * 0.5
        + win_rate * 0.6
        + pnl_pct * 0.8
        - (p95_ms / 1000.0) * 5.0
    )


def _write_ranked_candidates(rows: list[dict], out_dir: Path) -> None:
    rows_sorted = sorted(rows, key=lambda r: float(r.get("promotion_score", 0.0)), reverse=True)

    csv_file = out_dir / "promotion_candidates_ranked.csv"
    md_file = out_dir / "promotion_candidates_ranked.md"

    fieldnames = [
        "rank",
        "execution_profile",
        "run_dir",
        "passed_validation",
        "total_trades",
        "win_rate_pct",
        "gross_pnl_pct",
        "max_rolling_drawdown_pct",
        "max_consecutive_losses_observed",
        "p95_latency_ms",
        "promotion_score",
    ]

    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows_sorted, start=1):
            row_out = dict(row)
            row_out["rank"] = idx
            writer.writerow({k: row_out.get(k, "") for k in fieldnames})

    md_lines = [
        "# Phase 6b Promotion Candidates (Ranked)",
        "",
        "| Rank | Profile | Passed | Trades | Win Rate % | PnL % | Drawdown % | Max Loss Streak | P95 ms | Score |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows_sorted, start=1):
        md_lines.append(
            "| {rank} | {profile} | {passed} | {trades} | {wr:.2f} | {pnl:.2f} | {dd:.2f} | {ls} | {p95:.2f} | {score:.2f} |".format(
                rank=idx,
                profile=row.get("execution_profile", "unknown"),
                passed="yes" if row.get("passed_validation") else "no",
                trades=int(row.get("total_trades", 0) or 0),
                wr=float(row.get("win_rate_pct", 0.0) or 0.0),
                pnl=float(row.get("gross_pnl_pct", 0.0) or 0.0),
                dd=float(row.get("max_rolling_drawdown_pct", 0.0) or 0.0),
                ls=int(row.get("max_consecutive_losses_observed", 0) or 0),
                p95=float(row.get("p95_latency_ms", 0.0) or 0.0),
                score=float(row.get("promotion_score", 0.0) or 0.0),
            )
        )
    md_file.write_text("\n".join(md_lines), encoding="utf-8")


def _run_profile_batch(args, out_dir: Path) -> list[str]:
    profiles = _parse_profiles(args.profiles)
    if not profiles:
        raise SystemExit("No valid profiles found. Use scalper,moderate,conservative")

    report_files: list[str] = []
    batch_root = out_dir / "phase6b_profile_batch"
    batch_root.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        run_dir = batch_root / f"{profile}_{args.smoke_model}"
        rc = paper_rollout_main(
            [
                "--run-smoke",
                "--smoke-model", str(args.smoke_model),
                "--out-dir", str(run_dir),
                "--execution-profile", profile,
                "--max-iterations", str(args.max_iterations),
                "--poll-seconds", str(args.poll_seconds),
                "--mode", str(args.mode),
                "--qty", str(args.qty),
                "--base-capital", str(args.base_capital),
                "--target-win-rate-pct", str(args.target_win_rate_pct),
                "--min-trades-for-validation", str(args.min_trades_for_validation),
                "--max-consecutive-loss-streak", str(args.max_consecutive_loss_streak),
            ]
        )
        if int(rc) != 0:
            raise SystemExit(int(rc))

        report_file = run_dir / "paper_trading_validation.json"
        if report_file.exists():
            report_files.append(str(report_file))

    return report_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 6 tuning and promotion heuristics")
    parser.add_argument("--reports-glob", default="artifacts/backtests/**/paper_trading_validation.json", help="Glob path to validation reports")
    parser.add_argument("--out-dir", default="artifacts/phase6", help="Output directory for tuning artifacts")
    parser.add_argument("--min-reports-required", type=int, default=3, help="Minimum reports before hard gate enforcement")
    parser.add_argument("--run-profile-batch", action="store_true", help="Phase 6b: run profile batch (scalper/moderate/conservative) before tuning")
    parser.add_argument("--profiles", default="scalper,moderate,conservative", help="Comma-separated profiles for batch mode")
    parser.add_argument("--smoke-model", default="eurusd", help="Smoke model alias/path for batch mode")
    parser.add_argument("--max-iterations", type=int, default=3, help="Runner iterations for batch mode")
    parser.add_argument("--poll-seconds", type=float, default=0.2, help="Runner poll seconds for batch mode")
    parser.add_argument("--mode", default="dry_run_delayed", help="Runner mode for batch mode")
    parser.add_argument("--qty", type=float, default=0.01, help="Runner qty for batch mode")
    parser.add_argument("--base-capital", type=float, default=10000.0, help="Validation base capital for batch mode")
    parser.add_argument("--target-win-rate-pct", type=float, default=50.0, help="Validation target win rate for batch mode")
    parser.add_argument("--min-trades-for-validation", type=int, default=1, help="Validation min trades for batch mode")
    parser.add_argument("--max-consecutive-loss-streak", type=int, default=4, help="Validation max loss streak for batch mode")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_files: list[str]
    if args.run_profile_batch:
        report_files = _run_profile_batch(args, out_dir)
    else:
        report_files = sorted(glob(str(args.reports_glob), recursive=True))

    calibration = calibrate_promotion_gates(
        report_files,
        min_reports_required=int(args.min_reports_required),
    )
    tuning_plan = build_phase6_tuning_plan(report_files)

    calibration_out = out_dir / "promotion_gate_calibration.json"
    tuning_out = out_dir / "phase6_tuning_plan.json"
    save_promotion_gate_calibration(calibration, str(calibration_out))
    save_phase6_tuning_plan(tuning_plan, str(tuning_out))

    # Phase 6b ranking table
    ranking_rows: list[dict] = []
    for report_file in report_files:
        path = Path(report_file)
        report = json.loads(path.read_text(encoding="utf-8"))
        run_dir = str(path.parent)
        profile = "unknown"
        for p in ("scalper", "moderate", "conservative"):
            if p in run_dir.lower() or p in str(report.get("run_id", "")).lower():
                profile = p
                break
        row = {
            "execution_profile": profile,
            "run_dir": run_dir,
            "passed_validation": bool(report.get("passed_validation", False)),
            "total_trades": int(report.get("total_trades", 0) or 0),
            "win_rate_pct": float(report.get("win_rate_pct", 0.0) or 0.0),
            "gross_pnl_pct": float(report.get("gross_pnl_pct", 0.0) or 0.0),
            "max_rolling_drawdown_pct": float(report.get("max_rolling_drawdown_pct", 0.0) or 0.0),
            "max_consecutive_losses_observed": int(report.get("max_consecutive_losses_observed", 0) or 0),
            "p95_latency_ms": float(report.get("p95_latency_ms", 0.0) or 0.0),
        }
        row["promotion_score"] = _score_candidate(row)
        ranking_rows.append(row)

    if ranking_rows:
        _write_ranked_candidates(ranking_rows, out_dir)

    print(f"Reports scanned: {len(report_files)}")
    print(f"Wrote: {calibration_out}")
    print(f"Wrote: {tuning_out}")
    if ranking_rows:
        print(f"Wrote: {out_dir / 'promotion_candidates_ranked.csv'}")
        print(f"Wrote: {out_dir / 'promotion_candidates_ranked.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
