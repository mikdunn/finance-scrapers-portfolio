"""Phase 6b baseline test for profile batch optimizer and ranking outputs."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from projects.phase6_tuning import main as phase6_main


def test_phase6b_profile_batch_optimizer() -> bool:
    out_dir = Path("artifacts/backtests/backtest_phase6b")
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = phase6_main(
        [
            "--run-profile-batch",
            "--smoke-model", "eurusd",
            "--profiles", "scalper,moderate,conservative",
            "--out-dir", str(out_dir),
            "--max-iterations", "2",
            "--poll-seconds", "0.2",
            "--mode", "dry_run_delayed",
            "--min-reports-required", "3",
            "--min-trades-for-validation", "1",
            "--target-win-rate-pct", "50",
            "--max-consecutive-loss-streak", "4",
        ]
    )
    assert int(rc) == 0, f"phase6b command returned {rc}"

    rank_csv = out_dir / "promotion_candidates_ranked.csv"
    rank_md = out_dir / "promotion_candidates_ranked.md"
    cal_json = out_dir / "promotion_gate_calibration.json"
    plan_json = out_dir / "phase6_tuning_plan.json"

    assert rank_csv.exists(), "Missing promotion ranked CSV"
    assert rank_md.exists(), "Missing promotion ranked Markdown"
    assert cal_json.exists(), "Missing calibration JSON"
    assert plan_json.exists(), "Missing phase6 plan JSON"

    with rank_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 3, f"Expected at least 3 ranked candidates, got {len(rows)}"

    profiles = {row["execution_profile"] for row in rows}
    assert {"scalper", "moderate", "conservative"}.issubset(profiles), profiles

    # Verify sorted ranking by promotion score descending.
    scores = [float(row["promotion_score"]) for row in rows]
    assert scores == sorted(scores, reverse=True), scores

    print("=" * 70)
    print("PHASE 6B: PROFILE BATCH OPTIMIZER TEST")
    print("=" * 70)
    print(f"Candidates ranked: {len(rows)}")
    print(f"Top candidate: profile={rows[0]['execution_profile']} score={rows[0]['promotion_score']}")
    print(f"Artifacts: {rank_csv} | {rank_md} | {cal_json} | {plan_json}")
    print("✓ Phase 6b baseline test complete")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if test_phase6b_profile_batch_optimizer() else 1)
