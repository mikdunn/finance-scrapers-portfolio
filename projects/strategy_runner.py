"""Live/paper strategy runner scaffold.

This is a production-foundation runner, not a full broker OMS/EMS yet.
It provides:
- explicit runtime states (RUNNING/PAUSED/RISK_LOCK/HALTED)
- JSONL event journaling for replay/audit
- pre-trade and global kill-switch checks
- dry-run order intent emission from model signals
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import uuid

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

from utils.event_journal import EventJournal
from utils.broker_gateway import build_gateway, ExecutionPolicy, build_execution_policy
from utils.market_data import OHLCVRequest, fetch_ohlcv
from utils.ml_features import build_features
from utils.oms import OMS, OrderState
from utils.risk_controls import (
    RiskConfig, RiskState, 
    check_global_kill_switch, check_pretrade, check_in_trade, check_rejection_burst,
    record_rejection, record_trade_result
)
from utils.runtime_state import RunState, load_runtime_state, save_runtime_state


def _expected_feature_names(pipe) -> list[str] | None:
    for obj in (pipe, getattr(pipe, "named_steps", {}).get("imputer"), getattr(pipe, "named_steps", {}).get("model")):
        if obj is None:
            continue
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


def _align_features(X: pd.DataFrame, pipe) -> pd.DataFrame:
    names = _expected_feature_names(pipe)
    if not names:
        return X

    out = X.copy()
    for col in names:
        if col not in out.columns:
            out[col] = 0.0
    return out.loc[:, names]


def _signal_from_row(pipe, x_row: pd.DataFrame, *, signal_source: str, proba_enter: float, threshold: float) -> float:
    src = (signal_source or "predict").strip().lower()

    if src in {"proba", "predict_proba", "prob"} and hasattr(pipe, "predict_proba"):
        proba = np.asarray(pipe.predict_proba(x_row), dtype=float)
        model = getattr(pipe, "named_steps", {}).get("model")
        classes = np.asarray(getattr(model, "classes_", np.array([-1, 0, 1])))
        p_buy = None
        p_sell = None
        for i, c in enumerate(classes.tolist()):
            if int(c) == 1:
                p_buy = float(proba[0, i])
            elif int(c) == -1:
                p_sell = float(proba[0, i])

        sig = 0.0
        if p_buy is not None and p_buy >= float(proba_enter):
            sig = 1.0
        if p_sell is not None and p_sell >= float(proba_enter):
            sig = -1.0
        return sig

    pred = float(np.asarray(pipe.predict(x_row)).reshape(-1)[0])
    # Regression-style threshold fallback.
    if pred > float(threshold):
        return 1.0
    if pred < -float(threshold):
        return -1.0
    return float(np.clip(np.round(pred), -1, 1))


def _transition_state(state, state_file: str, journal: EventJournal, target: RunState, reason: str, extra: dict | None = None) -> None:
    state.set_state(target, reason)
    save_runtime_state(state_file, state)
    payload = {"state": state.state.value, "reason": state.reason}
    if extra:
        payload.update(extra)
    journal.append("state_transition", payload)


def _flatten_position_if_needed(*, symbol: str, px: float, oms: OMS, gateway, journal: EventJournal, execution_policy: ExecutionPolicy | None = None) -> None:
    cur = float(oms.net_position(symbol))
    if abs(cur) <= 1e-12:
        return

    side = "sell" if cur > 0 else "buy"
    qty = abs(cur)
    cid = str(uuid.uuid4())
    o = oms.create_order(client_order_id=cid, symbol=symbol, side=side, qty=qty, price_ref=px)
    journal.append("flatten_intent", {"symbol": symbol, "side": side, "qty": qty, "price_ref": px})
    
    if execution_policy is None:
        execution_policy = ExecutionPolicy()
    
    res = gateway.place_order(symbol=o.symbol, side=o.side, qty=o.qty, price_ref=o.price_ref, client_order_id=o.client_order_id, policy=execution_policy)
    journal.append(
        "flatten_submit",
        {
            "client_order_id": o.client_order_id,
            "broker_order_id": res.broker_order_id,
            "status": res.status,
            "reason": res.reason,
        },
    )
    status = str(res.status).lower()
    if status in {"rejected", "reject"}:
        oms.reject(o.client_order_id, reason=res.reason)
        return

    oms.ack(o.client_order_id, broker_order_id=res.broker_order_id)
    if float(res.filled_qty) > 0 and res.fill_price is not None:
        oms.fill(o.client_order_id, qty=float(res.filled_qty), price=float(res.fill_price), ts=res.ts)
        journal.append(
            "flatten_fill",
            {
                "client_order_id": o.client_order_id,
                "qty": float(res.filled_qty),
                "price": float(res.fill_price),
                "ts": str(res.ts),
            },
        )


def _poll_pending_orders(*, oms: OMS, gateway, journal: EventJournal) -> float:
    t0 = time.perf_counter()
    for o in list(oms.orders.values()):
        if o.state not in {OrderState.ACKED, OrderState.PARTIAL}:
            continue
        if not o.broker_order_id:
            continue

        upd = gateway.get_order_update(o.broker_order_id)
        status = str(upd.status).lower()
        journal.append(
            "order_update",
            {
                "client_order_id": o.client_order_id,
                "broker_order_id": o.broker_order_id,
                "status": status,
                "filled_qty": float(upd.filled_qty),
                "reason": upd.reason,
            },
        )

        if status in {"rejected", "reject"}:
            oms.reject(o.client_order_id, reason=upd.reason or "broker_reject")
            continue
        if status in {"canceled", "cancelled", "expired"}:
            oms.cancel(o.client_order_id, reason=status)
            continue

        if float(upd.filled_qty) > float(o.filled_qty) and upd.fill_price is not None:
            delta = float(upd.filled_qty) - float(o.filled_qty)
            oms.fill(o.client_order_id, qty=delta, price=float(upd.fill_price), ts=upd.ts)
            journal.append(
                "fill",
                {
                    "client_order_id": o.client_order_id,
                    "qty": float(delta),
                    "price": float(upd.fill_price),
                    "ts": str(upd.ts),
                    "source": "poll",
                },
            )
    return float((time.perf_counter() - t0) * 1000.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run live/paper signal loop with runtime states and risk gates")
    p.add_argument("--model", required=True, help="Path to trained model.joblib")
    p.add_argument("--symbol", default="BTC-USD", help="Single symbol for phase-1 runner")
    p.add_argument("--period", default="7d", help="Historical pull period to build latest feature row")
    p.add_argument("--interval", default="1m", help="Data interval")
    p.add_argument("--signal-source", default="proba", help="predict | proba")
    p.add_argument("--proba-enter", type=float, default=0.58, help="Probability threshold for entries")
    p.add_argument("--threshold", type=float, default=0.001, help="Regression threshold fallback")
    p.add_argument("--qty", type=float, default=0.01, help="Unit quantity per order intent")
    p.add_argument("--poll-seconds", type=float, default=5.0, help="Loop sleep between evaluations")
    p.add_argument("--max-iterations", type=int, default=30, help="Safety cap for iterations; 0 means infinite")

    p.add_argument("--state-file", default="artifacts/live/runtime_state.json", help="Runtime state file")
    p.add_argument("--events-file", default="artifacts/live/events.jsonl", help="Event journal JSONL file")
    p.add_argument("--metrics-file", default="artifacts/live/runner_metrics.json", help="Runner metrics output")

    p.add_argument("--start-state", default="RUNNING", help="RUNNING | PAUSED | RISK_LOCK | HALTED")
    p.add_argument("--max-notional-per-trade", type=float, default=2000.0)
    p.add_argument("--max-gross-notional", type=float, default=10000.0)
    p.add_argument("--hard-daily-loss-pct", type=float, default=2.0)
    p.add_argument("--max-consecutive-losses", type=int, default=4)
    p.add_argument("--reconcile-tolerance", type=float, default=1e-6, help="Allowed qty diff between OMS and broker position")
    p.add_argument("--max-stale-seconds", type=float, default=180.0, help="Max allowed age of latest market bar")

    p.add_argument("--execution-profile", default="scalper", help="scalper | moderate | conservative (execution strategy)")
    p.add_argument("--volatility-regime", type=float, default=1.0, help="Volatility multiplier for execution limits (1.0 = baseline)")

    p.add_argument("--mode", default="dry_run", help="dry_run | dry_run_delayed | alpaca")

    args = p.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    pipe = joblib.load(model_path)

    state = load_runtime_state(args.state_file)
    if not Path(args.state_file).exists():
        try:
            state.set_state(RunState(str(args.start_state).upper()), "startup")
        except Exception:
            state.set_state(RunState.HALTED, "invalid_start_state")
        save_runtime_state(args.state_file, state)

    journal = EventJournal(args.events_file)
    gateway = build_gateway(args.mode)
    execution_policy = build_execution_policy(
        profile=getattr(args, "execution_profile", "scalper"),
        volatility=float(getattr(args, "volatility_regime", 1.0))
    )
    oms = OMS()
    risk_cfg = RiskConfig(
        max_notional_per_trade=float(args.max_notional_per_trade),
        max_gross_notional=float(args.max_gross_notional),
        hard_daily_loss_pct=float(args.hard_daily_loss_pct),
        max_consecutive_losses=int(args.max_consecutive_losses),
    )
    risk_state = RiskState(start_equity=1.0, equity=1.0, gross_notional=0.0, consecutive_losses=0)
    last_px = 1.0

    lat_fetch_ms: list[float] = []
    lat_signal_ms: list[float] = []
    lat_order_ms: list[float] = []
    lat_poll_ms: list[float] = []
    lat_loop_ms: list[float] = []

    iterations = 0
    submitted = 0

    journal.append("runner_start", {"symbol": args.symbol, "mode": args.mode, "interval": args.interval})

    while True:
        loop_t0 = time.perf_counter()
        fetch_ms = 0.0
        signal_ms = 0.0
        order_ms = 0.0
        if int(args.max_iterations) > 0 and iterations >= int(args.max_iterations):
            break

        iterations += 1

        # Refresh state so operator edits to the state file are respected in-loop.
        state = load_runtime_state(args.state_file)

        poll_ms = _poll_pending_orders(oms=oms, gateway=gateway, journal=journal)
        lat_poll_ms.append(float(poll_ms))

        ok_global, reason_global = check_global_kill_switch(risk_cfg, risk_state)
        if not ok_global:
            _transition_state(state, args.state_file, journal, RunState.HALTED, reason_global)
            _flatten_position_if_needed(symbol=args.symbol, px=float(last_px), oms=oms, gateway=gateway, journal=journal)
            break

        if not state.can_submit_orders():
            # Safety action: if we're locked/halted externally, attempt flatten.
            if state.state in {RunState.RISK_LOCK, RunState.HALTED}:
                _flatten_position_if_needed(symbol=args.symbol, px=float(last_px), oms=oms, gateway=gateway, journal=journal)
            journal.append("loop_skip", {"state": state.state.value, "reason": state.reason})
            time.sleep(max(0.2, float(args.poll_seconds)))
            continue

        try:
            t_fetch0 = time.perf_counter()
            df = fetch_ohlcv(OHLCVRequest(symbol=args.symbol, period=args.period, interval=args.interval))
            fetch_ms = float((time.perf_counter() - t_fetch0) * 1000.0)
            lat_fetch_ms.append(fetch_ms)

            last_bar = pd.Timestamp(df.index[-1])
            if getattr(last_bar, "tzinfo", None) is not None:
                last_bar = last_bar.tz_convert("UTC").tz_localize(None)
            now_utc = pd.Timestamp.utcnow().tz_localize(None)
            age_seconds = float((now_utc - last_bar).total_seconds())
            journal.append("data_health", {"symbol": args.symbol, "age_seconds": age_seconds, "max_stale_seconds": float(args.max_stale_seconds)})
            if age_seconds > float(args.max_stale_seconds):
                _transition_state(state, args.state_file, journal, RunState.RISK_LOCK, "stale_data", extra={"age_seconds": age_seconds})
                _flatten_position_if_needed(symbol=args.symbol, px=float(last_px), oms=oms, gateway=gateway, journal=journal, execution_policy=execution_policy)
                continue
            
            # Phase 3: In-trade risk checks
            in_trade_ok, in_trade_reason = check_in_trade(risk_cfg, risk_state, data_age_seconds=age_seconds)
            if not in_trade_ok:
                _transition_state(state, args.state_file, journal, RunState.RISK_LOCK, in_trade_reason)
                _flatten_position_if_needed(symbol=args.symbol, px=float(last_px), oms=oms, gateway=gateway, journal=journal, execution_policy=execution_policy)
                continue

            X = build_features(df).dropna(how="all")
            if X.empty:
                journal.append("data_warning", {"reason": "no_features", "symbol": args.symbol})
                time.sleep(max(0.2, float(args.poll_seconds)))
                continue

            X = _align_features(X, pipe)
            row = X.tail(1)
            px = float(pd.to_numeric(df["Close"], errors="coerce").dropna().iloc[-1])
            last_px = px
            t_sig0 = time.perf_counter()
            signal = _signal_from_row(
                pipe,
                row,
                signal_source=str(args.signal_source),
                proba_enter=float(args.proba_enter),
                threshold=float(args.threshold),
            )
            signal_ms = float((time.perf_counter() - t_sig0) * 1000.0)
            lat_signal_ms.append(signal_ms)

            current_position = float(oms.net_position(args.symbol))
            desired_position = float(np.sign(signal) * abs(float(args.qty)))
            delta_qty = float(desired_position - current_position)

            if abs(delta_qty) > 0:
                ok_pretrade, reason_pretrade = check_pretrade(risk_cfg, risk_state, qty=delta_qty, px=px)
                if not ok_pretrade:
                    _transition_state(state, args.state_file, journal, RunState.RISK_LOCK, reason_pretrade)
                    _flatten_position_if_needed(symbol=args.symbol, px=px, oms=oms, gateway=gateway, journal=journal, execution_policy=execution_policy)
                    time.sleep(max(0.2, float(args.poll_seconds)))
                    continue

                side = "buy" if delta_qty > 0 else "sell"
                intent = {
                    "symbol": args.symbol,
                    "side": side,
                    "qty": abs(delta_qty),
                    "price_ref": px,
                    "mode": args.mode,
                    "reason": "signal_update",
                }
                journal.append("order_intent", intent)
                submitted += 1

                cid = str(uuid.uuid4())
                order = oms.create_order(
                    client_order_id=cid,
                    symbol=args.symbol,
                    side=side,
                    qty=abs(delta_qty),
                    price_ref=px,
                )
                t_ord0 = time.perf_counter()
                res = gateway.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    qty=order.qty,
                    price_ref=order.price_ref,
                    client_order_id=order.client_order_id,
                    policy=execution_policy,
                )
                order_ms = float((time.perf_counter() - t_ord0) * 1000.0)
                lat_order_ms.append(order_ms)
                journal.append(
                    "order_submit",
                    {
                        "client_order_id": order.client_order_id,
                        "broker_order_id": res.broker_order_id,
                        "status": res.status,
                        "reason": res.reason,
                    },
                )

                status = str(res.status).lower()
                if status in {"rejected", "reject"}:
                    oms.reject(order.client_order_id, reason=res.reason)
                    record_rejection(risk_state)
                    
                    # Check for rejection burst
                    burst_ok, burst_reason = check_rejection_burst(risk_cfg, risk_state)
                    if not burst_ok:
                        journal.append("rejection_burst_detected", {"rejection_burst": burst_reason})
                    
                    _transition_state(state, args.state_file, journal, RunState.RISK_LOCK, "order_rejected")
                    _flatten_position_if_needed(symbol=args.symbol, px=px, oms=oms, gateway=gateway, journal=journal, execution_policy=execution_policy)
                    continue

                oms.ack(order.client_order_id, broker_order_id=res.broker_order_id)

                if float(res.filled_qty) > 0 and res.fill_price is not None:
                    oms.fill(order.client_order_id, qty=float(res.filled_qty), price=float(res.fill_price), ts=res.ts)
                    journal.append(
                        "fill",
                        {
                            "client_order_id": order.client_order_id,
                            "qty": float(res.filled_qty),
                            "price": float(res.fill_price),
                            "ts": str(res.ts),
                        },
                    )

                current_position = float(oms.net_position(args.symbol))
                risk_state.gross_notional = abs(current_position * px)
                journal.append("position_update", {"symbol": args.symbol, "position": current_position, "gross_notional": risk_state.gross_notional})

            expected_qty = float(oms.net_position(args.symbol))
            broker_qty = float(gateway.get_position_qty(args.symbol))
            if abs(expected_qty - broker_qty) > float(args.reconcile_tolerance):
                _transition_state(
                    state,
                    args.state_file,
                    journal,
                    RunState.RISK_LOCK,
                    "position_mismatch",
                    extra={"expected_qty": expected_qty, "broker_qty": broker_qty},
                )
                _flatten_position_if_needed(symbol=args.symbol, px=px, oms=oms, gateway=gateway, journal=journal)
            else:
                journal.append("reconcile", {"expected_qty": expected_qty, "broker_qty": broker_qty})

            journal.append("signal", {"symbol": args.symbol, "signal": signal, "close": px, "state": state.state.value})

        except Exception as e:
            _transition_state(state, args.state_file, journal, RunState.RISK_LOCK, "runner_exception")
            journal.append("error", {"kind": "loop_exception", "error": str(e)})

        loop_ms = float((time.perf_counter() - loop_t0) * 1000.0)
        lat_loop_ms.append(loop_ms)
        journal.append(
            "loop_timing",
            {
                "fetch_ms": fetch_ms,
                "signal_ms": signal_ms,
                "order_ms": order_ms,
                "poll_ms": poll_ms,
                "loop_ms": loop_ms,
            },
        )

        time.sleep(max(0.2, float(args.poll_seconds)))

    def _lat(vals: list[float]) -> dict:
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()),
        }

    out = {
        "symbol": args.symbol,
        "mode": args.mode,
        "iterations": iterations,
        "submitted_order_intents": submitted,
        "final_position": float(oms.net_position(args.symbol)),
        "oms": oms.summary(),
        "latency_ms": {
            "fetch": _lat(lat_fetch_ms),
            "signal": _lat(lat_signal_ms),
            "order_submit": _lat(lat_order_ms),
            "poll": _lat(lat_poll_ms),
            "loop": _lat(lat_loop_ms),
        },
        "final_state": load_runtime_state(args.state_file).to_dict(),
    }
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    journal.append("runner_stop", out)

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {args.events_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
