"""Phase 5 baseline test for paper trading validation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from projects.paper_trading_rollout import main as paper_rollout_main


def test_phase5_validation_from_existing_artifacts() -> bool:
    out_dir = Path("artifacts/backtests/backtest_phase5")
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = paper_rollout_main(
        [
            "--metrics-file", "artifacts/final_integration_test/runner_metrics.json",
            "--events-file", "artifacts/final_integration_test/events.jsonl",
            "--out-dir", str(out_dir),
            "--symbol", "BTC-USD",
            "--broker", "paper_simulator",
            "--base-capital", "10000",
            "--min-trades-for-validation", "0",
            "--target-win-rate-pct", "50",
            "--max-consecutive-loss-streak", "10",
        ]
    )
    assert int(rc) == 0, f"paper rollout returned {rc}"

    json_path = out_dir / "paper_trading_validation.json"
    html_path = out_dir / "paper_trading_validation.html"
    assert json_path.exists(), "validation JSON was not created"
    assert html_path.exists(), "validation HTML was not created"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["symbol"] == "BTC-USD", data
    assert "total_trades" in data, data
    assert "max_consecutive_losses_observed" in data, data
    assert "p95_latency_ms" in data, data

    print("=" * 70)
    print("PHASE 5: PAPER TRADING VALIDATION TEST")
    print("=" * 70)
    print(f"Run ID: {data['run_id']}")
    print(f"Total trades: {data['total_trades']}")
    print(f"Win rate: {data['win_rate_pct']:.2f}%")
    print(f"Observed max loss streak: {data['max_consecutive_losses_observed']}")
    print(f"Validation passed: {data['passed_validation']}")
    print(f"P95 loop latency: {data['p95_latency_ms']:.2f}ms")
    print(f"Artifacts: {json_path} | {html_path}")
    print("✓ Phase 5 baseline test complete")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if test_phase5_validation_from_existing_artifacts() else 1)