"""Live Phase 3d: Trade Result Recording baseline test.

Tests trade closure detection, PnL calculation, and integration with risk engine.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import json

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.risk_controls import (
    RiskConfig, RiskState,
    detect_and_record_trade_close,
    record_trade_result,
    check_global_kill_switch,
)
from utils.event_journal import EventJournal


def test_trade_result_recording():
    """Test Phase 3d: Trade result recording for PnL tracking."""
    
    test_dir = Path("artifacts/backtests/backtest_phase3d")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    journal = EventJournal(str(test_dir / "events.jsonl"))
    risk_cfg = RiskConfig()
    risk_state = RiskState(start_equity=1.0, equity=1.0)
    
    journal.append("test_start", {
        "test": "phase3d_trade_result_recording",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    
    # Test scenarios: (symbol, position, price, expected_close, expected_profit)
    scenarios = [
        ("EURUSD", 100.0, 1.0800, False, None),      # Long entry
        ("EURUSD", 100.0, 1.0850, False, None),      # Hold
        ("EURUSD", 0.0, 1.0900, True, 0.00926),      # Close long with 0.926% profit
        
        ("AAPL", -50.0, 150.0, False, None),         # Short entry
        ("AAPL", -50.0, 149.0, False, None),         # Hold
        ("AAPL", 0.0, 148.0, True, 0.01333),         # Close short with 1.333% profit (150-148)/150
        
        ("TSLA", -100.0, 250.0, False, None),        # New short
        ("TSLA", -100.0, 249.0, False, None),        # Profit accruing
        ("TSLA", 0.0, 248.0, True, 0.008),           # Close short with 0.8% profit (250-248)/250
    ]
    
    results = []
    
    for symbol, position, price, should_close, expected_pnl in scenarios:
        was_closed, pnl_pct = detect_and_record_trade_close(
            risk_state=risk_state,
            symbol=symbol,
            current_position=position,
            current_price=price,
        )
        
        result = {
            "symbol": symbol,
            "position": position,
            "price": price,
            "was_closed": was_closed,
            "pnl_pct": pnl_pct,
            "pnl_bps": pnl_pct * 10000.0 if pnl_pct is not None else None,
        }
        results.append(result)
        
        # Verify expectations
        assert was_closed == should_close, f"{symbol}: Expected close={should_close}, got {was_closed}"
        
        if should_close:
            assert pnl_pct is not None, f"{symbol}: Expected PnL, got None"
            if expected_pnl:
                assert abs(pnl_pct - expected_pnl) < 0.0001, \
                    f"{symbol}: Expected PnL {expected_pnl:.4f}, got {pnl_pct:.4f}"
            
            # Update equity
            risk_state.equity *= (1.0 + pnl_pct)
        
        journal.append("trade_test_step", result)
    
    # Verify final state
    assert len(risk_state.recent_trades) == 3, f"Expected 3 trades, got {len(risk_state.recent_trades)}"
    
    total_pnl = sum(t['pnl'] for t in risk_state.recent_trades)
    expected_total = 0.00926 + 0.00678 + 0.008  # ~2.60%
    
    print("=" * 70)
    print("PHASE 3D: TRADE RESULT RECORDING TEST")
    print("=" * 70)
    print(f"✓ Trade closures detected: {len(risk_state.recent_trades)}")
    print(f"✓ Total PnL: {total_pnl:.2%}")
    print(f"✓ Equity growth: {(risk_state.equity - 1.0) * 100:.2f}%")
    print(f"✓ Consecutive loss tracking: {risk_state.consecutive_losses}")
    print()
    
    for i, trade in enumerate(risk_state.recent_trades, 1):
        print(f"  Trade {i}: {trade['pnl']:>7.2%}")
    
    print()
    
    # Verify kill-switch still operational
    kill_ok, kill_reason = check_global_kill_switch(risk_cfg, risk_state)
    assert kill_ok, f"Kill-switch should be ok, but got: {kill_reason}"
    print(f"✓ Global kill-switch operational: {kill_reason}")
    
    journal.append("test_complete", {
        "test": "phase3d_trade_result_recording",
        "trades_recorded": len(risk_state.recent_trades),
        "total_pnl": float(total_pnl),
        "final_equity": float(risk_state.equity),
        "status": "PASS",
    })
    
    print()
    print(f"✓ Phase 3d test complete - status: PASS")
    return True


if __name__ == "__main__":
    try:
        success = test_trade_result_recording()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
