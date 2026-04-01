"""Replay runner events and summarize deterministic outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.event_journal import EventJournal


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay JSONL runner events and summarize state transitions")
    p.add_argument("--events-file", required=True, help="Path to events JSONL")
    p.add_argument("--out", default="artifacts/live/replay_summary.json", help="Output summary JSON")
    args = p.parse_args(argv)

    ev = EventJournal(args.events_file).iter_events()

    final_state = "UNKNOWN"
    final_reason = ""
    counts: dict[str, int] = {}
    order_intents = 0
    errors = 0

    for e in ev:
        counts[e.kind] = counts.get(e.kind, 0) + 1
        if e.kind == "order_intent":
            order_intents += 1
        if e.kind == "error":
            errors += 1
        if e.kind == "state_transition":
            final_state = str(e.payload.get("state", final_state))
            final_reason = str(e.payload.get("reason", final_reason))
        if e.kind == "runner_stop":
            fs = e.payload.get("final_state")
            if isinstance(fs, dict):
                final_state = str(fs.get("state", final_state))
                final_reason = str(fs.get("reason", final_reason))

    out = {
        "events_file": str(args.events_file),
        "n_events": len(ev),
        "event_counts": counts,
        "order_intents": order_intents,
        "errors": errors,
        "final_state": final_state,
        "final_reason": final_reason,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
