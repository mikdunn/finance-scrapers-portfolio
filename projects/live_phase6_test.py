"""Phase 6 baseline test for tuning and promotion heuristics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from projects.phase6_tuning import main as phase6_main


def test_phase6_tuning_generation() -> bool:
    out_dir = Path("artifacts/backtests/backtest_phase6")
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = phase6_main(
        [
            "--reports-glob", "artifacts/backtests/**/paper_trading_validation.json",
            "--out-dir", str(out_dir),
            "--min-reports-required", "3",
        ]
    )
    assert int(rc) == 0, f"phase6 command returned {rc}"

    calibration_file = out_dir / "promotion_gate_calibration.json"
    plan_file = out_dir / "phase6_tuning_plan.json"
    assert calibration_file.exists(), "Missing calibration output"
    assert plan_file.exists(), "Missing phase6 plan output"

    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    assert "recommended_min_trades" in calibration, calibration
    assert "recommended_max_p95_latency_ms" in calibration, calibration
    assert "recommended_execution_profile" in plan, plan
    assert "priority_actions" in plan and isinstance(plan["priority_actions"], list), plan

    print("=" * 70)
    print("PHASE 6: TUNING AND PROMOTION HEURISTICS TEST")
    print("=" * 70)
    print(f"Reports used: {calibration['num_reports']}")
    print(f"Ready for hard enforcement: {calibration['ready_for_enforcement']}")
    print(f"Recommended min trades: {calibration['recommended_min_trades']}")
    print(f"Recommended min win rate: {calibration['recommended_min_win_rate_pct']:.2f}%")
    print(f"Recommended execution profile: {plan['recommended_execution_profile']}")
    print(f"Recommended poll seconds: {plan['recommended_poll_seconds']}")
    print("✓ Phase 6 baseline test complete")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if test_phase6_tuning_generation() else 1)
