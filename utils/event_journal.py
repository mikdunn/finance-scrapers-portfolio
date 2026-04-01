"""JSONL event journal for deterministic replay and auditing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    kind: str
    ts: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


class EventJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, payload: dict[str, Any]) -> None:
        ev = Event(kind=str(kind), ts=_utc_now_iso(), payload=payload)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(ev.to_json())
            f.write("\n")

    def iter_events(self) -> list[Event]:
        if not self.path.exists():
            return []

        out: list[Event] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(
                        Event(
                            kind=str(obj.get("kind", "unknown")),
                            ts=str(obj.get("ts", "")),
                            payload=dict(obj.get("payload", {})),
                        )
                    )
                except Exception:
                    continue
        return out
