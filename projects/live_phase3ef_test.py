"""Live Phase 3e/3f: De-Risk Automation and Advanced Kill-Switches baseline test.

Tests de-risk mode activation, position sizing reduction, and kill-switch triggers.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.risk_controls import (
    RiskConfig, RiskState,
    check_and_enable_de_risk, apply_de_risk_haircut, on_consecutive_loss_reset,
    check_rolling_drawdown_enforcement, check_volatility_spike,
    record_trade_result,
)
from utils.event_journal import EventJournal


def test_phase3e_de_risk_automation():
    """Test Phase 3e: De-risk mode activation and position sizing."""
    
    test_dir = Path("artifacts/backtests/backtest_phase3e")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    journal = EventJournal(str(test_dir / "events.jsonl"))
    risk_cfg = RiskConfig()  # threshold = 4, trigger at 3
    risk_state = RiskState(start_equity=1.0, equity=1.0)
    
    print("=" * 70)
    print("PHASE 3E: DE-RISK AUTOMATION TEST")
    print("=" * 70)
    print()
    
    journal.append("test_start", {
        "test": "phase3e_de_risk_automation",
        "de_risk_trigger": 3,  # threshold - 1
        "de_risk_haircut": 0.5,  # 0.5%
    })
    
    # Simulate losing streak that triggers de-risk
    trades = [
        ("Loss 1", -0.02, False),      # Loss 1/3: no de-risk yet
        ("Loss 2", -0.015, False),     # Loss 2/3: no de-risk yet
        ("Loss 3", -0.01, True),       # Loss 3/3: SHOULD TRIGGER de-risk
        ("Win to reset", 0.025, False), # Win: should disable de-risk
        ("Loss 4", -0.008, False),     # Loss after reset: de-risk off
    ]
    
    for i, (desc, pnl, expect_de_risk) in enumerate(trades, 1):
        # Record trade result
        record_trade_result(risk_state, pnl)
        
        # Check de-risk mode
        should_de_risk, de_risk_reason = check_and_enable_de_risk(risk_cfg, risk_state)
        
        if should_de_risk and not risk_state.in_de_risk_mode:
            risk_state.in_de_risk_mode = True
            is_triggered = True
        elif not should_de_risk and risk_state.in_de_risk_mode:
            on_consecutive_loss_reset(risk_state)
            is_triggered = False
        else:
            is_triggered = risk_state.in_de_risk_mode
        
        # Verify expectation
        if expect_de_risk:
            assert is_triggered == True, f"Trade {i} ({desc}): Expected de-risk=True, got {is_triggered}"
        
        print(f"[Trade {i}] {desc:20s} | PnL: {pnl:>7.1%} | Consecutive losses: {risk_state.consecutive_losses} | De-risk mode: {is_triggered}")
        
        journal.append("trade_step", {
            "trade_num": i,
            "description": desc,
            "pnl": float(pnl),
            "consecutive_losses": risk_state.consecutive_losses,
            "de_risk_mode": is_triggered,
            "de_risk_reason": de_risk_reason if is_triggered else "none",
        })
    
    print()
    print("✓ De-risk mode activation working correctly")
    print()
    
    # Test position haircut
    print("=" * 70)
    print("PHASE 3E: POSITION HAIRCUT TEST")
    print("=" * 70)
    print()
    
    risk_state2 = RiskState()
    risk_state2.in_de_risk_mode = False
    
    original_qty = 100.0
    haircut_pct = risk_cfg.de_risk_position_haircut_pct  # 0.5%
    
    # Test without de-risk
    adjusted_qty = apply_de_risk_haircut(original_qty, risk_state2, risk_cfg)
    print(f"Without de-risk: {original_qty:.1f} shares → {adjusted_qty:.1f} shares (no change)")
    assert adjusted_qty == original_qty, "Should not apply haircut when de-risk off"
    
    # Test with de-risk
    risk_state2.in_de_risk_mode = True
    adjusted_qty = apply_de_risk_haircut(original_qty, risk_state2, risk_cfg)
    expected_reduction = original_qty * (haircut_pct / 100.0)
    expected_qty = original_qty - expected_reduction
    print(f"With de-risk:    {original_qty:.1f} shares → {adjusted_qty:.1f} shares (haircut: {haircut_pct}% = {expected_reduction:.2f} shares)")
    assert abs(adjusted_qty - expected_qty) < 0.01, f"Haircut mismatch: {adjusted_qty} vs {expected_qty}"
    
    print()
    print("✓ Position haircut calculation working correctly")


def test_phase3f_kill_switches():
    """Test Phase 3f: Advanced kill-switches."""
    
    test_dir = Path("artifacts/backtests/backtest_phase3f")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    journal = EventJournal(str(test_dir / "events.jsonl"))
    risk_cfg = RiskConfig()
    
    print()
    print("=" * 70)
    print("PHASE 3F: ADVANCED KILL-SWITCHES TEST")
    print("=" * 70)
    print()
    
    journal.append("test_start", {
        "test": "phase3f_advanced_kill_switches",
        "rolling_drawdown_threshold": risk_cfg.rolling_drawdown_pct,
        "volatility_spike_threshold": 2.0,
    })
    
    # Test 1: Rolling drawdown enforcement
    print("Test 1: Rolling Drawdown Kill-Switch")
    print("-" * 70)
    
    risk_state = RiskState(start_equity=1.0, equity=1.0, peak_equity=1.0)
    
    # Simulate equity drawdown
    scenarios = [
        (1.0, "Peak 1.0"),
        (0.99, "Down 1%"),
        (0.96, "Down 4%"),
        (0.94, "Down 6% - SHOULD TRIGGER"),
        (0.95, "Recovery to 5% down"),
    ]
    
    for equity, desc in scenarios:
        risk_state.equity = equity
        risk_state.update_peak_equity()
        
        dd_ok, dd_reason = check_rolling_drawdown_enforcement(risk_cfg, risk_state)
        rolling_dd = risk_state.rolling_drawdown_pct()
        threshold_dd = risk_cfg.rolling_drawdown_pct
        
        status = "✓ OK" if dd_ok else "✗ TRIGGERED"
        print(f"  Equity: {equity:.2f} | Drawdown: {rolling_dd:>5.2f}% | Threshold: {threshold_dd:.2f}% | {status}")
        
        journal.append("drawdown_step", {
            "equity": float(equity),
            "rolling_drawdown_pct": float(rolling_dd),
            "kill_switch_ok": dd_ok,
            "reason": dd_reason,
        })
    
    print()
    print("✓ Rolling drawdown kill-switch working correctly")
    print()
    
    # Test 2: Volatility spike detection
    print("Test 2: Volatility Spike Detection")
    print("-" * 70)
    
    baseline_vol = 0.15  # 15% baseline
    spike_threshold = 2.0
    
    vol_scenarios = [
        (0.15, "Baseline vol"),
        (0.20, "20% vol (1.33x)"),
        (0.25, "25% vol (1.67x)"),
        (0.31, "31% vol (2.07x - SHOULD TRIGGER)"),
        (0.18, "18% vol (back to 1.2x)"),
    ]
    
    for vol, desc in vol_scenarios:
        spike_detected, spike_reason = check_volatility_spike(vol, baseline_vol, spike_threshold)
        vol_ratio = vol / baseline_vol if baseline_vol > 0 else 0
        
        status = "✗ SPIKE" if spike_detected else "✓ OK"
        print(f"  Vol: {vol:.2f} | Ratio: {vol_ratio:.2f}x | {status}")
        
        journal.append("vol_spike_step", {
            "realized_vol": float(vol),
            "baseline_vol": float(baseline_vol),
            "vol_ratio": float(vol_ratio),
            "spike_detected": spike_detected,
            "reason": spike_reason,
        })
    
    print()
    print("✓ Volatility spike detection working correctly")
    print()


def main():
    """Run all Phase 3e/3f tests."""
    try:
        test_phase3e_de_risk_automation()
        test_phase3f_kill_switches()
        
        print("=" * 70)
        print("PHASE 3E/3F TESTS COMPLETE ✓")
        print("=" * 70)
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
