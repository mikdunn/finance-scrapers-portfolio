"""Live risk controls for paper/canary strategy runner.

Phase 3 enhancement: Real-time risk engine with pre-trade and in-trade checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk limits and checks."""
    # Pre-trade limits
    max_notional_per_trade: float = 2000.0
    max_gross_notional: float = 10000.0
    max_net_notional: float = 5000.0
    max_concurrent_positions: int = 10
    max_order_rate_per_minute: int = 50
    
    # In-trade limits
    hard_daily_loss_pct: float = 2.0
    rolling_drawdown_pct: float = 5.0  # Max peak-to-trough within session
    max_consecutive_losses: int = 4
    max_rejection_burst: int = 3  # Max rejections in rolling window
    rejection_window_seconds: float = 10.0
    max_stale_data_seconds: float = 60.0
    max_slippage_drift_bps: float = 50.0
    
    # De-risking
    de_risk_position_haircut_pct: float = 0.5  # Reduce position size by 0.5% under stress


@dataclass
class RiskState:
    """Mutable state tracking for in-trade risk engine."""
    start_equity: float = 1.0
    equity: float = 1.0
    peak_equity: float = 1.0  # For drawdown calculation
    gross_notional: float = 0.0
    net_notional: float = 0.0
    consecutive_losses: int = 0
    
    # Rejection tracking for burst detection
    recent_rejections: list[float] = field(default_factory=list)  # Timestamps of rejections
    
    # Slippage tracking
    recent_slippages: list[float] = field(default_factory=list)  # Slippage values (bps)
    
    # Trade history for consecutive-loss detection
    recent_trades: list[dict] = field(default_factory=list)  # [{"pnl": float, "ts": str}, ...]
    
    # Order tracking
    orders_submitted_minute: int = 0
    last_order_time: Optional[float] = None
    
    # Health status
    in_de_risk_mode: bool = False
    stale_data_detected: bool = False

    def daily_loss_pct(self) -> float:
        if self.start_equity <= 0:
            return 0.0
        return max(0.0, (self.start_equity - self.equity) / self.start_equity * 100.0)
    
    def rolling_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity * 100.0)
    
    def update_peak_equity(self) -> None:
        """Update peak equity for drawdown calculation."""
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity


def check_pretrade(risk_cfg: RiskConfig, risk_state: RiskState, *, qty: float, px: float, num_concurrent_positions: int = 0) -> tuple[bool, str]:
    """Pre-trade checks before placing an order.
    
    Args:
        risk_cfg: Risk configuration
        risk_state: Current risk state
        qty: Order quantity
        px: Price reference
        num_concurrent_positions: Current number of open positions
    
    Returns:
        (allowed: bool, reason: str)
    """
    notional = abs(float(qty) * float(px))
    
    if notional > float(risk_cfg.max_notional_per_trade):
        return False, "max_notional_per_trade"

    if float(risk_state.gross_notional) + notional > float(risk_cfg.max_gross_notional):
        return False, "max_gross_notional"
    
    if int(num_concurrent_positions) >= int(risk_cfg.max_concurrent_positions):
        return False, "max_concurrent_positions"
    
    # Order rate check
    now = datetime.now(timezone.utc).timestamp()
    if risk_state.last_order_time is not None:
        time_since_last = now - risk_state.last_order_time
        if time_since_last < 1.0:  # Same second
            # In real implementation, track per-minute rolling window
            pass
    
    return True, "ok"


def check_in_trade(risk_cfg: RiskConfig, risk_state: RiskState, data_age_seconds: float) -> tuple[bool, str]:
    """In-trade checks during active trading.
    
    Returns:
        (allow_continues: bool, reason: str)
    """
    # Check rolling drawdown
    if risk_state.rolling_drawdown_pct() >= float(risk_cfg.rolling_drawdown_pct):
        return False, "rolling_drawdown"
    
    # Check data staleness
    if data_age_seconds > float(risk_cfg.max_stale_data_seconds):
        return False, "stale_data"
    
    return True, "ok"


def check_global_kill_switch(risk_cfg: RiskConfig, risk_state: RiskState) -> tuple[bool, str]:
    """Global kill-switch: hard daily loss or streak violations.
    
    Returns:
        (allow_continues: bool, reason: str)
    """
    if risk_state.daily_loss_pct() >= float(risk_cfg.hard_daily_loss_pct):
        return False, "hard_daily_loss"

    if int(risk_state.consecutive_losses) >= int(risk_cfg.max_consecutive_losses):
        return False, "max_consecutive_losses"

    return True, "ok"


def check_rejection_burst(risk_cfg: RiskConfig, risk_state: RiskState) -> tuple[bool, str]:
    """Detect rejection bursts (rapid-fire rejections).
    
    Returns:
        (allow_continues: bool, reason: str)
    """
    now = datetime.now(timezone.utc).timestamp()
    window_start = now - float(risk_cfg.rejection_window_seconds)
    
    # Purge old rejections
    risk_state.recent_rejections = [ts for ts in risk_state.recent_rejections if ts > window_start]
    
    if len(risk_state.recent_rejections) >= int(risk_cfg.max_rejection_burst):
        return False, "rejection_burst"
    
    return True, "ok"


def record_rejection(risk_state: RiskState) -> None:
    """Record a new order rejection."""
    risk_state.recent_rejections.append(datetime.now(timezone.utc).timestamp())


def record_trade_result(risk_state: RiskState, pnl: float, ts: Optional[str] = None) -> None:
    """Record trade outcome and update consecutive loss counter.
    
    Args:
        risk_state: Risk state to update
        pnl: Trade PnL (positive = win, negative = loss)
        ts: Timestamp (ISO string), defaults to now
    """
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    
    is_loss = float(pnl) < 0
    
    if is_loss:
        risk_state.consecutive_losses += 1
    else:
        risk_state.consecutive_losses = 0  # Reset on win
    
    risk_state.recent_trades.append({"pnl": float(pnl), "ts": str(ts), "is_loss": is_loss})
    
    # Keep only recent trades (e.g., last 100)
    if len(risk_state.recent_trades) > 100:
        risk_state.recent_trades = risk_state.recent_trades[-100:]


def record_slippage(risk_state: RiskState, slippage_bps: float) -> None:
    """Record slippage observation for drift detection."""
    risk_state.recent_slippages.append(float(slippage_bps))
    if len(risk_state.recent_slippages) > 50:
        risk_state.recent_slippages = risk_state.recent_slippages[-50:]

