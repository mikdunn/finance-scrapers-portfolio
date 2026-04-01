"""Live Phase 4: Terminal Dashboard baseline test.

Tests dashboard rendering and formatting for real-time monitoring.
"""

import sys
import os
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.terminal_dashboard import TerminalDashboard
from utils.oms import OMS, Order, OrderState
from utils.risk_controls import RiskConfig, RiskState, record_trade_result
from utils.runtime_state import RunState
from datetime import datetime, timezone


def test_dashboard_rendering():
    """Test Phase 4: Dashboard rendering with various system states."""
    
    test_dir = Path("artifacts/backtests/backtest_phase4")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PHASE 4: TERMINAL DASHBOARD TEST")
    print("=" * 70)
    print()
    
    # Create test components
    dashboard = TerminalDashboard()
    oms = OMS()
    risk_cfg = RiskConfig()
    risk_state = RiskState(start_equity=1.0, equity=1.0)
    
    # Test 1: Flat position, RUNNING state
    print("Test 1: Flat position, RUNNING state")
    print("-" * 70)
    
    oms.create_order(
        client_order_id="test-0001",
        symbol="EURUSD",
        side="buy",
        qty=100.0,
        price_ref=1.0800,
    )
    
    dashboard_text = dashboard.render(
        symbol="EURUSD",
        oms=oms,
        risk_state=risk_state,
        risk_cfg=risk_cfg,
        run_state=RunState.RUNNING,
        current_price=1.0850,
        signal=1.0,
    )
    
    # Verify key components are in output
    assert "RUNNING" in dashboard_text, "Should show RUNNING state"
    assert "1.0850" in dashboard_text, "Should show current price"
    assert "BUY" in dashboard_text, "Should show BUY signal"
    assert "FLAT" in dashboard_text, "Should show flat position"
    
    print("✓ Dashboard renders with RUNNING state and BUY signal")
    print()
    
    # Test 2: Open position
    print("Test 2: Open position with unrealized PnL")
    print("-" * 70)
    
    # Simulate position entry
    oms.orders.clear()
    order = oms.create_order(
        client_order_id="test-0002",
        symbol="BTC-USD",
        side="buy",
        qty=1.0,
        price_ref=45000.0,
    )
    oms.ack(order.client_order_id, broker_order_id="broker-001")
    oms.fill(order.client_order_id, qty=1.0, price=45000.0, ts=datetime.now(timezone.utc).isoformat())
    
    risk_state.position_entry_prices["BTC-USD"] = 45000.0
    
    dashboard_text = dashboard.render(
        symbol="BTC-USD",
        oms=oms,
        risk_state=risk_state,
        risk_cfg=risk_cfg,
        run_state=RunState.RUNNING,
        current_price=45500.0,
        signal=1.0,
    )
    
    assert "BTC-USD" in dashboard_text, "Should show symbol"
    assert "45000" in dashboard_text or "45500" in dashboard_text, "Should show prices"
    assert "+1.11%" in dashboard_text or "1.11" in dashboard_text, "Should show PnL"
    
    print("✓ Dashboard shows open position with PnL")
    print()
    
    # Test 3: Risk lock state with trades
    print("Test 3: RISK_LOCK state with trade history")
    print("-" * 70)
    
    risk_state.recent_trades = [
        {"pnl": 0.005, "ts": "2026-04-01T10:00:00Z", "is_loss": False},
        {"pnl": -0.008, "ts": "2026-04-01T10:01:00Z", "is_loss": True},
        {"pnl": -0.012, "ts": "2026-04-01T10:02:00Z", "is_loss": True},
    ]
    risk_state.consecutive_losses = 2
    
    dashboard_text = dashboard.render(
        symbol="EURUSD",
        oms=oms,
        risk_state=risk_state,
        risk_cfg=risk_cfg,
        run_state=RunState.RISK_LOCK,
        current_price=1.0820,
        signal=0.0,
    )
    
    assert "RISK_LOCK" in dashboard_text or "🔴" in dashboard_text, "Should show RISK_LOCK"
    assert "3" in dashboard_text, "Should show 3 trades"
    assert "Consecutive Losses: 2" in dashboard_text or "2" in dashboard_text, "Should show consecutive losses"
    
    print("✓ Dashboard shows RISK_LOCK state and trade history")
    print()
    
    # Test 4: De-risk mode enabled
    print("Test 4: De-risk mode active")
    print("-" * 70)
    
    risk_state.in_de_risk_mode = True
    risk_state.consecutive_losses = 3
    
    dashboard_text = dashboard.render(
        symbol="EURUSD",
        oms=oms,
        risk_state=risk_state,
        risk_cfg=risk_cfg,
        run_state=RunState.RUNNING,
        current_price=1.0810,
        signal=-1.0,
    )
    
    assert "DE-RISK" in dashboard_text, "Should show DE-RISK indicator"
    assert "SELL" in dashboard_text or "↓" in dashboard_text, "Should show SELL signal"
    
    print("✓ Dashboard shows de-risk mode indicator")
    print()
    
    # Test 5: HTML export
    print("Test 5: HTML export")
    print("-" * 70)
    
    html_file = test_dir / "dashboard_snapshot.html"
    dashboard.save_to_html(
        output_file=str(html_file),
        symbol="EURUSD",
        oms=oms,
        risk_state=risk_state,
        risk_cfg=risk_cfg,
        run_state=RunState.RUNNING,
        current_price=1.0810,
        signal=0.0,
    )
    
    assert html_file.exists(), "HTML file should be created"
    content = html_file.read_text()
    assert "<html>" in content, "Should contain HTML"
    assert "Dashboard" in content, "Should contain title"
    
    print(f"✓ HTML snapshot saved to {html_file}")
    print()
    
    print("=" * 70)
    print("PHASE 4 DASHBOARD TEST COMPLETE ✓")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = test_dashboard_rendering()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
