"""Phase 4: Terminal Dashboard for real-time monitoring.

Provides live view of trading system state, risk metrics, and performance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from utils.oms import OMS, OrderState
from utils.risk_controls import RiskConfig, RiskState
from utils.runtime_state import RunState


class TerminalDashboard:
    """Real-time terminal dashboard for trading system monitoring."""
    
    def __init__(self, height: int = 30):
        """Initialize dashboard.
        
        Args:
            height: Number of lines to display (default 30 for standard terminal)
        """
        self.height = height
    
    def render(
        self,
        symbol: str,
        oms: OMS,
        risk_state: RiskState,
        risk_cfg: RiskConfig,
        run_state: RunState,
        current_price: Optional[float] = None,
        signal: Optional[float] = None,
    ) -> str:
        """Render dashboard as formatted string.
        
        Args:
            symbol: Trading symbol
            oms: Order management system
            risk_state: Current risk state
            risk_cfg: Risk configuration
            run_state: Runner state
            current_price: Current market price
            signal: Current signal (-1, 0, or 1)
        
        Returns:
            Formatted dashboard string ready for display
        """
        lines = []
        
        # Header
        now_utc = datetime.now(timezone.utc).isoformat()[:19]
        lines.append(f"╔═══════════════════════════════════════════════════════════════╗")
        lines.append(f"║ BROKER-GRADE SCALPING PLATFORM - PHASE 4 DASHBOARD           ║")
        lines.append(f"║ {now_utc} UTC                                       ║")
        lines.append(f"╚═══════════════════════════════════════════════════════════════╝")
        lines.append("")
        
        # System Status
        state_color = self._state_marker(run_state)
        lines.append(f"┌─ SYSTEM STATUS ─────────────────────────────────────────────────┐")
        lines.append(f"│ State: {state_color} {run_state.value:14s} │ Symbol: {symbol:8s}                       │")
        lines.append(f"│ PID: {Path('/proc/self/stat').read_text().split()[0]:40s} │")
        lines.append(f"└─────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Market Data
        if current_price is not None:
            signal_str = self._signal_str(signal)
            lines.append(f"┌─ MARKET DATA ───────────────────────────────────────────────────┐")
            lines.append(f"│ Price: ${current_price:12.4f}  │  Signal: {signal_str:^18s} │")
            lines.append(f"└─────────────────────────────────────────────────────────────────┘")
            lines.append("")
        
        # Position Status
        position = oms.net_position(symbol)
        position_str = f"{position:+8.0f}" if position != 0 else "       FLAT"
        position_color = "🟢" if position == 0 else ("🔴" if abs(position) > 100 else "🟡")
        
        lines.append(f"┌─ POSITION ──────────────────────────────────────────────────────┐")
        lines.append(f"│ {position_color} Position: {position_str} shares                              │")
        
        # Entry price if position exists
        if position != 0 and symbol in risk_state.position_entry_prices:
            entry_px = risk_state.position_entry_prices[symbol]
            if entry_px > 0 and current_price:
                unrealized_pnl = (current_price - entry_px) / entry_px if position > 0 else (entry_px - current_price) / entry_px
                pnl_str = f"{unrealized_pnl:+.2%}"
                lines.append(f"│ Entry Price: ${entry_px:12.4f}  │  Unrealized PnL: {pnl_str:^10s} │")
        
        lines.append(f"└─────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Risk Metrics
        equity = risk_state.equity
        peak_eq = risk_state.peak_equity
        dd_pct = risk_state.rolling_drawdown_pct()
        daily_loss_pct = risk_state.daily_loss_pct()
        
        lines.append(f"┌─ RISK METRICS ──────────────────────────────────────────────────┐")
        lines.append(f"│ Equity: ${equity:.4f}              │ Peak: ${peak_eq:.4f}                   │")
        lines.append(f"│ Daily Loss: {daily_loss_pct:>6.2f}%             │ Rolling DD: {dd_pct:>6.2f}%              │")
        
        # Risk status indicators
        dd_ok = dd_pct < risk_cfg.rolling_drawdown_pct
        loss_ok = daily_loss_pct < risk_cfg.hard_daily_loss_pct
        dd_marker = "✓" if dd_ok else "✗"
        loss_marker = "✓" if loss_ok else "✗"
        
        lines.append(f"│ {dd_marker} Drawdown OK (threshold: {risk_cfg.rolling_drawdown_pct:.1f}%)    │ {loss_marker} Daily Loss OK (threshold: {risk_cfg.hard_daily_loss_pct:.1f}%)   │")
        lines.append(f"└─────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Trade Statistics
        num_trades = len(risk_state.recent_trades)
        consecutive_losses = risk_state.consecutive_losses
        in_de_risk = risk_state.in_de_risk_mode
        
        lines.append(f"┌─ TRADE STATISTICS ──────────────────────────────────────────────┐")
        lines.append(f"│ Trades Recorded: {num_trades:3d}              │ Wins: {self._count_wins(risk_state):3d}              │")
        lines.append(f"│ Consecutive Losses: {consecutive_losses:2d}        │ Losses: {self._count_losses(risk_state):3d}              │")
        
        de_risk_marker = "(DE-RISK)" if in_de_risk else "         "
        lines.append(f"│ {de_risk_marker}                                              │")
        lines.append(f"└─────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Orders Status
        oms_summary = oms.summary()
        lines.append(f"┌─ ORDERS ────────────────────────────────────────────────────────┐")
        
        order_line = f"│ Total: {oms_summary['n_orders']:3d}  │ "
        state_counts = oms_summary.get('state_counts', {})
        for state_name in [OrderState.NEW.value, OrderState.ACKED.value, OrderState.FILLED.value]:
            count = state_counts.get(state_name, 0)
            order_line += f"{state_name}: {count:2d}  │ "
        
        order_line = order_line.rstrip(" │") + "│"
        lines.append(order_line)
        lines.append(f"└─────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Footer
        lines.append(f"ℹ️  Phase 4 Dashboard | Phase 3 Risk Engine Active | Real-time Monitoring")
        
        return "\n".join(lines)
    
    def _state_marker(self, state: RunState) -> str:
        """Get colored marker for run state."""
        if state == RunState.RUNNING:
            return "🟢"
        elif state == RunState.PAUSED:
            return "🟡"
        elif state == RunState.RISK_LOCK:
            return "🔴"
        else:  # HALTED
            return "⛔"
    
    def _signal_str(self, signal: Optional[float]) -> str:
        """Format signal for display."""
        if signal is None:
            return "NEUTRAL"
        elif signal > 0.5:
            return "BUY (↑)"
        elif signal < -0.5:
            return "SELL (↓)"
        else:
            return "NEUTRAL (–)"
    
    def _count_wins(self, risk_state: RiskState) -> int:
        """Count winning trades."""
        return sum(1 for t in risk_state.recent_trades if t.get('pnl', 0) > 0)
    
    def _count_losses(self, risk_state: RiskState) -> int:
        """Count losing trades."""
        return sum(1 for t in risk_state.recent_trades if t.get('pnl', 0) < 0)
    
    def save_to_html(
        self,
        output_file: str,
        symbol: str,
        oms: OMS,
        risk_state: RiskState,
        risk_cfg: RiskConfig,
        run_state: RunState,
        current_price: Optional[float] = None,
        signal: Optional[float] = None,
    ) -> None:
        """Export dashboard snapshot to HTML file.
        
        Useful for creating archived dashboard reports.
        """
        html = f"""
        <html>
        <head>
            <title>Trading Dashboard - {datetime.now(timezone.utc).isoformat()}</title>
            <style>
                body {{ font-family: monospace; background: #1e1e1e; color: #00ff00; padding: 20px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
                .running {{ color: #00ff00; }}
                .paused {{ color: #ffff00; }}
                .risk-lock {{ color: #ff6600; }}
                .halted {{ color: #ff0000; }}
            </style>
        </head>
        <body>
            <h1>Broker-Grade Scalping Platform Dashboard</h1>
            <p>Snapshot: {datetime.now(timezone.utc).isoformat()}</p>
            <pre>
            {self.render(symbol, oms, risk_state, risk_cfg, run_state, current_price, signal)}
            </pre>
        </body>
        </html>
        """
        
        Path(output_file).write_text(html)
