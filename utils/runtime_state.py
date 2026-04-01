"""Runtime state manager for live/paper strategy processes.

Defines explicit run states and a tiny persisted state store.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any


class RunState(str, Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RISK_LOCK = "RISK_LOCK"
    HALTED = "HALTED"


@dataclass
class RuntimeStateStore:
    state: RunState = RunState.PAUSED
    reason: str = "startup"

    def can_submit_orders(self) -> bool:
        return self.state == RunState.RUNNING

    def set_state(self, state: RunState, reason: str) -> None:
        self.state = state
        self.reason = (reason or "").strip() or "unspecified"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["state"] = str(self.state.value)
        return out


def load_runtime_state(path: str | Path) -> RuntimeStateStore:
    p = Path(path)
    if not p.exists():
        return RuntimeStateStore()

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return RuntimeStateStore(state=RunState.HALTED, reason="state_file_corrupt")

    raw_state = str(obj.get("state", RunState.PAUSED.value)).upper()
    reason = str(obj.get("reason", "loaded"))

    try:
        state = RunState(raw_state)
    except Exception:
        state = RunState.HALTED
        reason = "unknown_state_value"

    return RuntimeStateStore(state=state, reason=reason)


def save_runtime_state(path: str | Path, state: RuntimeStateStore) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
