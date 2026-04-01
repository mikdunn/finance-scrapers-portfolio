"""Live runner event monitor with threshold-based health evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.event_journal import EventJournal


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate runner event logs against health thresholds")
    p.add_argument("--events-file", required=True, help="Path to runner events JSONL")
    p.add_argument("--out", default="artifacts/live/monitor_report.json", help="Output monitor report JSON")
    p.add_argument("--max-error-events", type=int, default=0)
    p.add_argument("--max-loop-p95-ms", type=float, default=1000.0)
    p.add_argument("--max-stale-seconds", type=float, default=180.0)
    args = p.parse_args(argv)

    events = EventJournal(args.events_file).iter_events()
    counts: dict[str, int] = {}
    errors = 0
    transitions: list[dict] = []
    loop_ms: list[float] = []
    fetch_ms: list[float] = []
    signal_ms: list[float] = []
    order_ms: list[float] = []
    data_age_s: list[float] = []

    for e in events:
        counts[e.kind] = counts.get(e.kind, 0) + 1
        if e.kind == "error":
            errors += 1
        if e.kind == "state_transition":
            transitions.append(e.payload)
        if e.kind == "loop_timing":
            loop_ms.append(float(e.payload.get("loop_ms", float("nan"))))
            fetch_ms.append(float(e.payload.get("fetch_ms", float("nan"))))
            signal_ms.append(float(e.payload.get("signal_ms", float("nan"))))
            order_ms.append(float(e.payload.get("order_ms", float("nan"))))
        if e.kind == "data_health":
            data_age_s.append(float(e.payload.get("age_seconds", float("nan"))))

    breaches: list[str] = []
    if errors > int(args.max_error_events):
        breaches.append(f"error_events>{args.max_error_events}")

    loop_p95 = _pct([x for x in loop_ms if np.isfinite(x)], 95.0)
    if np.isfinite(loop_p95) and loop_p95 > float(args.max_loop_p95_ms):
        breaches.append(f"loop_p95_ms>{args.max_loop_p95_ms}")

    latest_age = float("nan")
    finite_age = [x for x in data_age_s if np.isfinite(x)]
    if finite_age:
        latest_age = float(finite_age[-1])
        if latest_age > float(args.max_stale_seconds):
            breaches.append(f"latest_data_age_s>{args.max_stale_seconds}")

    report = {
        "events_file": str(args.events_file),
        "n_events": len(events),
        "counts": counts,
        "errors": errors,
        "transitions": transitions,
        "latency_ms": {
            "loop_p50": _pct([x for x in loop_ms if np.isfinite(x)], 50.0),
            "loop_p95": loop_p95,
            "fetch_p95": _pct([x for x in fetch_ms if np.isfinite(x)], 95.0),
            "signal_p95": _pct([x for x in signal_ms if np.isfinite(x)], 95.0),
            "order_p95": _pct([x for x in order_ms if np.isfinite(x)], 95.0),
        },
        "latest_data_age_s": latest_age,
        "thresholds": {
            "max_error_events": int(args.max_error_events),
            "max_loop_p95_ms": float(args.max_loop_p95_ms),
            "max_stale_seconds": float(args.max_stale_seconds),
        },
        "breaches": breaches,
        "ok": len(breaches) == 0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
