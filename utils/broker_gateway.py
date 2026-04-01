"""Broker gateway abstraction with dry-run and Alpaca adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Protocol
import uuid

import requests


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GatewayResult:
    status: str
    broker_order_id: str | None = None
    filled_qty: float = 0.0
    fill_price: float | None = None
    reason: str = ""
    ts: str = ""


@dataclass
class ExecutionPolicy:
    """Execution policy configuration for orders (scalping-oriented)."""
    order_type: str = "market"  # market, limit
    time_in_force: str = "day"  # day, ioc, fok, gtc
    limit_offset_bps: float = 0.0  # For limit orders: offset from price_ref in basis points
    timeout_seconds: float = 60.0  # Cancel if unfilled after this many seconds
    
    def __post_init__(self):
        self.order_type = (self.order_type or "market").strip().lower()
        self.time_in_force = (self.time_in_force or "day").strip().lower()
        if self.order_type not in {"market", "limit", "stop"}:
            self.order_type = "market"
        if self.time_in_force not in {"day", "ioc", "fok", "gtc"}:
            self.time_in_force = "day"


class BrokerGateway(Protocol):
    def place_order(self, *, symbol: str, side: str, qty: float, price_ref: float, client_order_id: str, policy: ExecutionPolicy | None = None) -> GatewayResult:
        ...

    def get_order_update(self, broker_order_id: str) -> GatewayResult:
        ...

    def cancel_order(self, broker_order_id: str) -> GatewayResult:
        ...

    def get_position_qty(self, symbol: str) -> float:
        ...


class DryRunGateway:
    def __init__(self, *, delayed_fill: bool = False):
        self._positions: dict[str, float] = {}
        self._delayed_fill = bool(delayed_fill)
        self._pending: dict[str, dict[str, float | str]] = {}

    def place_order(self, *, symbol: str, side: str, qty: float, price_ref: float, client_order_id: str, policy: ExecutionPolicy | None = None) -> GatewayResult:
        if policy is None:
            policy = ExecutionPolicy()
        
        # For IOC/FOK in dry-run: immediate fill or rejection
        if policy.time_in_force in {"ioc", "fok"}:
            # In dry-run, we always fill immediately
            signed = abs(float(qty)) if str(side).lower() == "buy" else -abs(float(qty))
            self._positions[str(symbol)] = self._positions.get(str(symbol), 0.0) + signed
            return GatewayResult(
                status="filled",
                broker_order_id=f"dry-{client_order_id}",
                filled_qty=abs(float(qty)),
                fill_price=float(price_ref),
                reason=f"dry_run_ioc_fill:{policy.time_in_force}",
                ts=_utc_now_iso(),
            )
        
        # For delayed fill (standard paper trading)
        if self._delayed_fill:
            broker_order_id = f"dry-{client_order_id}"
            self._pending[broker_order_id] = {
                "symbol": str(symbol),
                "side": str(side).lower(),
                "qty": abs(float(qty)),
                "price_ref": float(price_ref),
            }
            return GatewayResult(
                status="accepted",
                broker_order_id=broker_order_id,
                filled_qty=0.0,
                fill_price=None,
                reason="dry_run_delayed_fill",
                ts=_utc_now_iso(),
            )

        # Immediate fill for standard dry-run
        signed = abs(float(qty)) if str(side).lower() == "buy" else -abs(float(qty))
        self._positions[str(symbol)] = self._positions.get(str(symbol), 0.0) + signed
        return GatewayResult(
            status="filled",
            broker_order_id=f"dry-{client_order_id}",
            filled_qty=abs(float(qty)),
            fill_price=float(price_ref),
            reason="dry_run_immediate_fill",
            ts=_utc_now_iso(),
        )

    def get_order_update(self, broker_order_id: str) -> GatewayResult:
        if str(broker_order_id) in self._pending:
            rec = self._pending.pop(str(broker_order_id))
            side = str(rec["side"])
            qty = float(rec["qty"])
            px = float(rec["price_ref"])
            symbol = str(rec["symbol"])
            signed = qty if side == "buy" else -qty
            self._positions[symbol] = self._positions.get(symbol, 0.0) + signed
            return GatewayResult(
                status="filled",
                broker_order_id=str(broker_order_id),
                filled_qty=qty,
                fill_price=px,
                reason="dry_run_delayed_fill_complete",
                ts=_utc_now_iso(),
            )
        return GatewayResult(status="filled", broker_order_id=str(broker_order_id), ts=_utc_now_iso())

    def cancel_order(self, broker_order_id: str) -> GatewayResult:
        self._pending.pop(str(broker_order_id), None)
        return GatewayResult(status="canceled", broker_order_id=str(broker_order_id), ts=_utc_now_iso())

    def get_position_qty(self, symbol: str) -> float:
        return float(self._positions.get(str(symbol), 0.0))


class AlpacaGateway:
    def __init__(self, *, base_url: str | None = None, key_id: str | None = None, secret: str | None = None, timeout: float = 10.0):
        self.base_url = (base_url or os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
        self.key_id = key_id or os.getenv("ALPACA_API_KEY_ID", "")
        self.secret = secret or os.getenv("ALPACA_API_SECRET_KEY", "")
        self.timeout = float(timeout)

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        }

    def place_order(self, *, symbol: str, side: str, qty: float, price_ref: float, client_order_id: str, policy: ExecutionPolicy | None = None) -> GatewayResult:
        if policy is None:
            policy = ExecutionPolicy()
        
        if not self.key_id or not self.secret:
            return GatewayResult(status="rejected", reason="alpaca_credentials_missing", ts=_utc_now_iso())

        url = f"{self.base_url}/v2/orders"
        
        # Determine order type and parameters
        order_type = policy.order_type
        tif = policy.time_in_force
        
        payload = {
            "symbol": str(symbol),
            "side": str(side).lower(),
            "type": order_type,
            "time_in_force": tif,
            "qty": str(abs(float(qty))),
            "client_order_id": str(client_order_id),
        }
        
        # For limit orders, compute limit price from price_ref and offset
        if order_type == "limit":
            limit_offset = float(policy.limit_offset_bps) / 10000.0
            if str(side).lower() == "buy":
                # For buys, limit price is slightly lower (tighter spread)
                limit_px = float(price_ref) * (1.0 - abs(limit_offset))
            else:
                # For sells, limit price is slightly higher (tighter spread) 
                limit_px = float(price_ref) * (1.0 + abs(limit_offset))
            payload["limit_price"] = str(round(limit_px, 8))
        
        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
            if r.status_code >= 300:
                return GatewayResult(status="rejected", reason=f"alpaca_http_{r.status_code}", ts=_utc_now_iso())
            data = r.json() if r.text else {}
            order_id = str(data.get("id", uuid.uuid4()))
            status = str(data.get("status", "accepted")).lower()
            return GatewayResult(status=status, broker_order_id=order_id, ts=_utc_now_iso())
        except Exception as e:
            return GatewayResult(status="rejected", reason=f"alpaca_exception:{e}", ts=_utc_now_iso())

    def get_order_update(self, broker_order_id: str) -> GatewayResult:
        if not self.key_id or not self.secret:
            return GatewayResult(status="rejected", reason="alpaca_credentials_missing", ts=_utc_now_iso())

        url = f"{self.base_url}/v2/orders/{broker_order_id}"
        try:
            r = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if r.status_code >= 300:
                return GatewayResult(status="rejected", broker_order_id=str(broker_order_id), reason=f"alpaca_http_{r.status_code}", ts=_utc_now_iso())
            data = r.json() if r.text else {}
            status = str(data.get("status", "accepted")).lower()
            filled_qty = float(data.get("filled_qty", 0.0) or 0.0)
            fill_price_raw = data.get("filled_avg_price")
            fill_price = float(fill_price_raw) if fill_price_raw not in {None, ""} else None
            return GatewayResult(
                status=status,
                broker_order_id=str(data.get("id", broker_order_id)),
                filled_qty=filled_qty,
                fill_price=fill_price,
                ts=_utc_now_iso(),
            )
        except Exception as e:
            return GatewayResult(status="rejected", broker_order_id=str(broker_order_id), reason=f"alpaca_exception:{e}", ts=_utc_now_iso())

    def cancel_order(self, broker_order_id: str) -> GatewayResult:
        if not self.key_id or not self.secret:
            return GatewayResult(status="rejected", reason="alpaca_credentials_missing", ts=_utc_now_iso())

        url = f"{self.base_url}/v2/orders/{broker_order_id}"
        try:
            r = requests.delete(url, headers=self._headers(), timeout=self.timeout)
            if r.status_code >= 300:
                return GatewayResult(status="rejected", broker_order_id=str(broker_order_id), reason=f"alpaca_http_{r.status_code}", ts=_utc_now_iso())
            return GatewayResult(status="canceled", broker_order_id=str(broker_order_id), ts=_utc_now_iso())
        except Exception as e:
            return GatewayResult(status="rejected", broker_order_id=str(broker_order_id), reason=f"alpaca_exception:{e}", ts=_utc_now_iso())

    def get_position_qty(self, symbol: str) -> float:
        if not self.key_id or not self.secret:
            return 0.0

        url = f"{self.base_url}/v2/positions/{symbol}"
        try:
            r = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if r.status_code == 404:
                return 0.0
            if r.status_code >= 300:
                return 0.0
            data = r.json() if r.text else {}
            return float(data.get("qty", 0.0))
        except Exception:
            return 0.0



def build_execution_policy(*, profile: str = "scalper", volatility: float = 1.0) -> ExecutionPolicy:
    """Build execution policy based on scalping profile and volatility regime.
    
    Args:
        profile: "scalper" (tight, fast), "moderate" (balanced), "conservative" (patient)
        volatility: volatility multiplier (1.0 = baseline), affects limit offset
    
    Returns:
        ExecutionPolicy configured for the given conditions
    """
    v = float(volatility)
    p = (profile or "scalper").strip().lower()
    
    if p in {"scalper", "scalp", "tight"}:
        # Ultra-tight execution for scalpers: IOC, no limit offset (market-like)
        return ExecutionPolicy(
            order_type="market",
            time_in_force="ioc",
            limit_offset_bps=0.0,
            timeout_seconds=5.0,  # Fast timeout for scalping
        )
    elif p in {"moderate", "med", "balanced"}:
        # Balanced: FOK limit orders with small offset, adjusts for volatility
        return ExecutionPolicy(
            order_type="limit",
            time_in_force="fok",
            limit_offset_bps=1.0 * v,  # 1 bps base, scales with volatility
            timeout_seconds=15.0,
        )
    else:  # conservative
        # Patient limit orders with larger offset, longer timeout
        return ExecutionPolicy(
            order_type="limit",
            time_in_force="gtc",
            limit_offset_bps=3.0 * v,  # 3 bps base, scales with volatility
            timeout_seconds=60.0,
        )


def build_gateway(mode: str) -> BrokerGateway:
    m = (mode or "dry_run").strip().lower()
    if m in {"alpaca", "alpaca_paper", "paper_alpaca"}:
        return AlpacaGateway()
    if m in {"dry_run_delayed", "dry-delayed", "dry_pending"}:
        return DryRunGateway(delayed_fill=True)
    return DryRunGateway()
