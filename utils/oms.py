"""Minimal in-process order management state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderState(str, Enum):
    NEW = "NEW"
    ACKED = "ACKED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Fill:
    qty: float
    price: float
    ts: str


@dataclass
class Order:
    client_order_id: str
    symbol: str
    side: str
    qty: float
    price_ref: float
    state: OrderState = OrderState.NEW
    broker_order_id: str | None = None
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    reason: str = ""
    fills: list[Fill] = field(default_factory=list)

    def remaining_qty(self) -> float:
        return max(0.0, float(self.qty) - float(self.filled_qty))


class OMS:
    def __init__(self) -> None:
        self.orders: dict[str, Order] = {}

    def create_order(self, *, client_order_id: str, symbol: str, side: str, qty: float, price_ref: float) -> Order:
        o = Order(
            client_order_id=str(client_order_id),
            symbol=str(symbol),
            side=str(side).lower(),
            qty=abs(float(qty)),
            price_ref=float(price_ref),
        )
        self.orders[o.client_order_id] = o
        return o

    def ack(self, client_order_id: str, *, broker_order_id: str | None = None) -> None:
        o = self.orders[client_order_id]
        o.state = OrderState.ACKED
        o.broker_order_id = broker_order_id

    def reject(self, client_order_id: str, *, reason: str) -> None:
        o = self.orders[client_order_id]
        o.state = OrderState.REJECTED
        o.reason = str(reason)

    def cancel(self, client_order_id: str, *, reason: str = "") -> None:
        o = self.orders[client_order_id]
        o.state = OrderState.CANCELED
        o.reason = str(reason)

    def fill(self, client_order_id: str, *, qty: float, price: float, ts: str) -> None:
        o = self.orders[client_order_id]
        q = abs(float(qty))
        p = float(price)
        if q <= 0:
            return

        prev_filled = float(o.filled_qty)
        new_filled = min(float(o.qty), prev_filled + q)
        delta = new_filled - prev_filled
        if delta <= 0:
            return

        if o.avg_fill_price is None:
            o.avg_fill_price = p
        else:
            o.avg_fill_price = (o.avg_fill_price * prev_filled + p * delta) / new_filled

        o.filled_qty = new_filled
        o.fills.append(Fill(qty=delta, price=p, ts=str(ts)))

        if o.filled_qty >= float(o.qty):
            o.state = OrderState.FILLED
        else:
            o.state = OrderState.PARTIAL

    def net_position(self, symbol: str) -> float:
        sym = str(symbol)
        pos = 0.0
        for o in self.orders.values():
            if o.symbol != sym:
                continue
            if o.state not in {OrderState.PARTIAL, OrderState.FILLED}:
                continue
            side = o.side.lower()
            if side == "buy":
                pos += float(o.filled_qty)
            elif side == "sell":
                pos -= float(o.filled_qty)
        return float(pos)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for o in self.orders.values():
            counts[o.state.value] = counts.get(o.state.value, 0) + 1
        return {
            "n_orders": len(self.orders),
            "state_counts": counts,
        }
