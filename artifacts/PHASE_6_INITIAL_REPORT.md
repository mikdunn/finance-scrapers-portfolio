# Phase 6: Performance Tuning and Promotion Heuristics (Initial)

Date: April 1, 2026
Status: Started

## Scope Delivered

This initial Phase 6 increment introduces a repeatable tuning/heuristics workflow driven by Phase 5 paper-trading validation artifacts.

Implemented components:

1. Phase 6 CLI: `projects/phase6_tuning.py`
2. Gate calibration model in `utils/paper_trading_validation.py`
3. Tuning plan generation model in `utils/paper_trading_validation.py`
4. Baseline test: `projects/live_phase6_test.py`
5. Top-level route: `main.py --project phase6`
6. Phase 6b profile-batch optimizer with ranked candidates
7. Baseline test: `projects/live_phase6b_test.py`

## Inputs and Outputs

Inputs:

1. `paper_trading_validation.json` files from Phase 5 runs
2. Optional report glob filter

Outputs:

1. `promotion_gate_calibration.json`
2. `phase6_tuning_plan.json`

## Current Calibration Snapshot

Generated from 5 paper validation reports:

1. Recommended min trades: 5
2. Recommended min win rate: 50.0%
3. Recommended max consecutive loss streak: 2
4. Recommended max rolling drawdown: 2.0%
5. Recommended max p95 loop latency: 796.89 ms

Note: all reports currently have zero closed trades in this environment, so win-rate calibration remains provisional.

## Current Phase 6 Heuristics Snapshot

1. Recommended execution profile: scalper
2. Recommended poll seconds: 0.878
3. Recommended max iterations per smoke pass: 25
4. Suggested hard daily loss: 2.0%
5. Suggested rolling drawdown threshold: 3.0%
6. Suggested max consecutive losses: 4

Priority actions emitted:

1. Keep p95 loop latency under 1 second
2. Run side-by-side profile comparison
3. Apply calibrated gates before canary enablement

## Verification

1. `python -m projects.live_phase6_test` passes
2. `python main.py --project phase6 ...` produces artifacts
3. No diagnostics errors in changed files

## Next Increment

1. Generate non-zero closed-trade reports from longer paper runs
2. Add profile-aware calibration using execution profile metadata
3. Add optimizer loop to tune poll interval and risk thresholds automatically
4. Connect Phase 6 outputs into canary promotion gating for Phase 7

## Phase 6b Update

Phase 6b is now implemented as an execution-profile batch mode in `projects/phase6_tuning.py`:

1. Runs fresh smoke paper validations for `scalper`, `moderate`, and `conservative`
2. Produces calibrated gate outputs from the generated reports
3. Emits ranked candidate artifacts:
	- `promotion_candidates_ranked.csv`
	- `promotion_candidates_ranked.md`

This enables direct profile comparison and promotion candidate ranking from one command.
