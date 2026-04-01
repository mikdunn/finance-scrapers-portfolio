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
    # Position tracking for PnL calculation (Phase 3d)
    position_entry_prices: dict[str, float] = field(default_factory=dict)  # {symbol: avg_entry_price}
    position_entry_qty: dict[str, float] = field(default_factory=dict)  # {symbol: qty_at_entry}
    last_position: dict[str, float] = field(default_factory=dict)  # {symbol: last_position_qty} for detecting exits
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


def detect_and_record_trade_close(
    risk_state: RiskState,
    symbol: str,
    current_position: float,
    current_price: float,
    ts: Optional[str] = None,
) -> tuple[bool, Optional[float]]:
    """Detect and record trade closures when position changes direction or closes.
    
    Phase 3d: When a position closes (transitions through zero), calculate PnL and record.
    
    Args:
        risk_state: Risk state to update
        symbol: Trading symbol
        current_position: Current net position qty
        current_price: Current market price
        ts: Timestamp (ISO string), defaults to now
    
    Returns:
        (was_closed: bool, pnl_pct: Optional[float])
        - was_closed: True if a position closed in this update
        - pnl_pct: Realized PnL percentage if closed, else None
    """
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    
    symbol_str = str(symbol)
    prev_position = float(risk_state.last_position.get(symbol_str, 0.0))
    entry_price = float(risk_state.position_entry_prices.get(symbol_str, 0.0))
    
    # Initialize on first call
    if symbol_str not in risk_state.last_position:
        risk_state.last_position[symbol_str] = 0.0
    
    # Update position tracking
    risk_state.last_position[symbol_str] = float(current_position)
    
    # Check if position transitioned to zero or changed sign (closed)
    prev_sign = 1.0 if prev_position > 0 else (-1.0 if prev_position < 0 else 0.0)
    curr_sign = 1.0 if current_position > 0 else (-1.0 if current_position < 0 else 0.0)
    
    was_closed = False
    pnl_pct = None
    
    # Detect closure: position was open and now is zero, or direction changed
    if prev_position != 0.0 and (float(current_position) == 0.0 or prev_sign != curr_sign):
        was_closed = True
        
        # Calculate PnL: (exit_price - entry_price) / entry_price
        if entry_price > 0:
            if prev_position > 0:
                # Was long, now flat/short: (sell_price - buy_price) / buy_price
                pnl_pct = (float(current_price) - entry_price) / entry_price
            else:
                # Was short, now flat/long: (short_price - cover_price) / short_price
                pnl_pct = (entry_price - float(current_price)) / entry_price
            
            # Record the trade result
            record_trade_result(risk_state, pnl_pct, ts=ts)
        
        # Reset entry price tracking
        risk_state.position_entry_prices[symbol_str] = 0.0
        risk_state.position_entry_qty[symbol_str] = 0.0
    
    # Update entry price when position increases (new entry)
    elif float(current_position) != 0.0 and abs(float(current_position)) > abs(prev_position):
        # Position increased (adding to existing or new position)
        if symbol_str not in risk_state.position_entry_prices or risk_state.position_entry_prices[symbol_str] == 0.0:
            # New position
            risk_state.position_entry_prices[symbol_str] = float(current_price)
            risk_state.position_entry_qty[symbol_str] = abs(float(current_position))
        else:
            # Average in
            prev_qty = float(risk_state.position_entry_qty.get(symbol_str, 0.0))
            prev_price = float(risk_state.position_entry_prices.get(symbol_str, 0.0))
            new_qty = abs(float(current_position))
            delta_qty = new_qty - prev_qty
            
            if delta_qty > 0 and prev_price > 0:
                # Average the entry price
                avg_price = (prev_price * prev_qty + float(current_price) * delta_qty) / new_qty
                risk_state.position_entry_prices[symbol_str] = avg_price
                risk_state.position_entry_qty[symbol_str] = new_qty
    
    return was_closed, pnl_pct


def check_and_enable_de_risk(risk_cfg: RiskConfig, risk_state: RiskState) -> tuple[bool, str]:
    """Check if de-risk mode should be enabled based on consecutive losses.
    
    Phase 3e: Trigger de-risk when consecutive losses reach threshold - 1.
    This allows recovery opportunity before hard kill-switch triggers.
    
    Returns:
        (should_enable_de_risk: bool, reason: str)
    """
    trigger_threshold = int(risk_cfg.max_consecutive_losses) - 1
    
    if int(risk_state.consecutive_losses) >= trigger_threshold:
        return True, f"consecutive_losses_{risk_state.consecutive_losses}_exceeds_{trigger_threshold}"
    
    return False, "ok"


def apply_de_risk_haircut(qty: float, risk_state: RiskState, risk_cfg: RiskConfig) -> float:
    """Apply position size haircut under de-risk mode.
    
    Phase 3e: Reduce position size by de_risk_position_haircut_pct.
    Default: 0.5% reduction per trade while in de-risk mode.
    
    Args:
        qty: Original position size (absolute value)
        risk_state: Current risk state
        risk_cfg: Risk configuration with haircut percentage
    
    Returns:
        Adjusted position size with haircut applied
    """
    if not risk_state.in_de_risk_mode:
        return float(qty)
    
    haircut_pct = float(risk_cfg.de_risk_position_haircut_pct) / 100.0
    reduction = float(qty) * haircut_pct
    adjusted_qty = float(qty) - reduction
    
    return max(0.0, adjusted_qty)


def on_consecutive_loss_reset(risk_state: RiskState) -> None:
    """Exit de-risk mode when a winning trade is recorded.
    
    Phase 3e: Called when consecutive_losses resets to 0 after a winning trade.
    """
    if risk_state.in_de_risk_mode:
        risk_state.in_de_risk_mode = False


def check_rolling_drawdown_enforcement(risk_cfg: RiskConfig, risk_state: RiskState) -> tuple[bool, str]:
    """Enforce rolling drawdown kill-switch.
    
    Phase 3f: Implements rolling drawdown defense (peak-to-trough).
    When rolling drawdown exceeds threshold, trigger RISK_LOCK and auto-flatten.
    
    Returns:
        (allow_continue: bool, reason: str)
    """
    rolling_dd_pct = risk_state.rolling_drawdown_pct()
    threshold_pct = float(risk_cfg.rolling_drawdown_pct)
    
    if rolling_dd_pct >= threshold_pct:
        return False, f"rolling_drawdown_{rolling_dd_pct:.2f}%_exceeds_{threshold_pct:.2f}%"
    
    return True, "ok"


def check_volatility_spike(realized_vol: float, baseline_vol: float, spike_threshold: float = 2.0) -> tuple[bool, str]:
    """Detect volatility spikes for position auto-reduction.
    
    Phase 3f: Advanced kill-switch for market regime changes.
    If realized_vol > spike_threshold * baseline_vol, recommend position reduction.
    
    Args:
        realized_vol: Current realized volatility
        baseline_vol: Historical baseline volatility
        spike_threshold: Multiplier threshold (default 2.0 = 2x)
    
    Returns:
        (spike_detected: bool, reason: str)
    """
    if baseline_vol <= 0:
        return False, "no_baseline"
    
    vol_ratio = float(realized_vol) / float(baseline_vol)
    
    if vol_ratio >= float(spike_threshold):
        return True, f"vol_spike_{vol_ratio:.2f}x_baseline"
    
    return False, "ok"

