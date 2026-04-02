"""Phase 5: Paper Trading Validation & Rollout Framework.

Provides infrastructure for running the broker-grade scalping platform against
paper trading (sandbox) environments to validate all phases before live trading.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading validation runs."""
    
    # Broker configuration
    broker: str = "alpaca"  # alpaca | paper_simulator
    paper_trading_account: str = "default"
    base_capital: float = 10000.0  # Starting cash for paper account
    
    # Trading parameters
    symbol: str = "EURUSD"
    execution_profile: str = "scalper"  # scalper | moderate | conservative
    volatility_regime: float = 1.0  # 1.0 = baseline
    
    # Risk parameters (inherit from Phase 3)
    max_notional_per_trade: float = 2000.0
    max_gross_notional: float = 10000.0
    hard_daily_loss_pct: float = 2.0
    max_consecutive_losses: int = 4
    rolling_drawdown_pct: float = 5.0
    de_risk_position_haircut_pct: float = 0.5
    
    # Validation parameters
    min_trades_for_validation: int = 20  # Minimum trades before considering valid
    target_win_rate_pct: float = 55.0  # Target win rate % for validation
    max_consecutive_loss_streak: int = 3  # Max allowed consecutive losses
    
    # Monitoring
    dashboard_enabled: bool = True
    log_interval_seconds: float = 300.0  # Log summary every 5 minutes
    
    # Validation duration
    validation_duration_days: int = 5  # Run for 5 days of market data
    max_iterations: int = 0  # 0 = no limit, run until validation complete


@dataclass
class ValidationResult:
    """Results from a paper trading validation run."""
    
    # Run metadata
    run_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    symbol: str
    broker: str
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_consecutive_losses_observed: int
    win_rate_pct: float
    
    # PnL metrics
    starting_cash: float
    final_cash: float
    gross_pnl: float
    gross_pnl_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    
    # Risk metrics
    max_daily_loss_pct: float
    max_rolling_drawdown_pct: float
    peak_equity: float
    final_equity: float
    
    # Risk engine metrics
    rejections_encountered: int
    de_risk_mode_activations: int
    kill_switch_triggers: int
    
    # Validation status
    passed_validation: bool
    validation_errors: list[str]
    recommendations: list[str]
    
    # Latency metrics
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float


@dataclass
class PromotionGateCalibration:
    """Calibrated promotion gates from paper-trading validation history."""

    num_reports: int
    min_reports_required: int
    ready_for_enforcement: bool
    source_files: list[str]
    recommended_min_trades: int
    recommended_min_win_rate_pct: float
    recommended_max_consecutive_loss_streak: int
    recommended_max_rolling_drawdown_pct: float
    recommended_max_p95_latency_ms: float
    notes: list[str]


@dataclass
class Phase6TuningPlan:
    """Phase 6 tuning plan derived from paper-trading validation history."""

    num_reports: int
    profile_counts: dict[str, int]
    recommended_execution_profile: str
    recommended_poll_seconds: float
    recommended_max_iterations: int
    suggested_hard_daily_loss_pct: float
    suggested_rolling_drawdown_pct: float
    suggested_max_consecutive_losses: int
    priority_actions: list[str]


class PaperTradingValidator:
    """Validator for paper trading runs."""
    
    def __init__(self, config: PaperTradingConfig):
        """Initialize validator with configuration."""
        self.config = config
        self.run_id = self._generate_run_id()
        self.validation_errors: list[str] = []
        self.recommendations: list[str] = []
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        now = datetime.now(timezone.utc)
        return f"papertrading_{now.strftime('%Y%m%d_%H%M%S')}"
    
    def validate_config(self) -> bool:
        """Validate paper trading configuration.
        
        Returns:
            True if config is valid, False otherwise
        """
        self.validation_errors = []
        
        # Check broker
        if self.config.broker not in {"alpaca", "paper_simulator"}:
            self.validation_errors.append(f"Invalid broker: {self.config.broker}")
        
        # Check symbol
        if not self.config.symbol or len(self.config.symbol) < 2:
            self.validation_errors.append(f"Invalid symbol: {self.config.symbol}")
        
        # Check capital
        if self.config.base_capital <= 0:
            self.validation_errors.append(f"Invalid base capital: {self.config.base_capital}")
        
        # Check execution profile
        if self.config.execution_profile not in {"scalper", "moderate", "conservative"}:
            self.validation_errors.append(f"Invalid execution profile: {self.config.execution_profile}")
        
        # Check risk parameters
        if self.config.max_consecutive_losses < 2:
            self.validation_errors.append("max_consecutive_losses should be >= 2")
        
        if self.config.rolling_drawdown_pct <= 0 or self.config.rolling_drawdown_pct > 20:
            self.validation_errors.append(f"rolling_drawdown_pct should be 0-20%")
        
        # Check validation parameters
        if self.config.target_win_rate_pct < 50 or self.config.target_win_rate_pct > 100:
            self.validation_errors.append(f"target_win_rate_pct should be 50-100%")
        
        return len(self.validation_errors) == 0
    
    def check_validation_criteria(self, result: ValidationResult) -> bool:
        """Check if paper trading results meet validation criteria.
        
        Args:
            result: ValidationResult from paper trading run
        
        Returns:
            True if validation passed, False otherwise
        """
        self.validation_errors = []
        self.recommendations = []
        
        # Minimum trade count
        if result.total_trades < self.config.min_trades_for_validation:
            self.validation_errors.append(
                f"Insufficient trades: {result.total_trades} < {self.config.min_trades_for_validation}"
            )
        
        # Win rate check
        if result.win_rate_pct < self.config.target_win_rate_pct:
            self.validation_errors.append(
                f"Win rate too low: {result.win_rate_pct:.1f}% < {self.config.target_win_rate_pct}%"
            )
            self.recommendations.append(
                "Review signal generation model - may need retraining"
            )
        
        # Consecutive loss streak check
        if result.max_consecutive_losses_observed > self.config.max_consecutive_loss_streak:
            self.validation_errors.append(
                "Consecutive loss streak too high: "
                f"{result.max_consecutive_losses_observed} > {self.config.max_consecutive_loss_streak}"
            )
            self.recommendations.append(
                "Consider tightening de-risk thresholds or adding volatility filter"
            )
        
        # Drawdown check
        if result.max_rolling_drawdown_pct > self.config.rolling_drawdown_pct:
            self.validation_errors.append(
                f"Rolling drawdown exceeded: {result.max_rolling_drawdown_pct:.1f}% > {self.config.rolling_drawdown_pct}%"
            )
            self.recommendations.append(
                "Reduce position size or tighten rolling drawdown threshold"
            )
        
        # Kill switch triggers
        if result.kill_switch_triggers > 5:
            self.recommendations.append(
                f"High kill-switch trigger count ({result.kill_switch_triggers}) - check market conditions"
            )
        
        # Overall profitability (allow zero or negative for initial validation)
        if result.total_trades > 0:
            if result.gross_pnl_pct < -5:
                self.recommendations.append(
                    f"Significant loss ({result.gross_pnl_pct:.1f}%) - review risk parameters"
                )
        
        # Set passed_validation based on critical criteria
        passed = len(self.validation_errors) == 0
        result.passed_validation = passed
        result.validation_errors = self.validation_errors
        result.recommendations = self.recommendations
        
        return passed


def load_jsonl_events(events_file: str) -> list[dict[str, Any]]:
    """Load JSONL event records from disk."""
    path = Path(events_file)
    if not path.exists():
        raise FileNotFoundError(str(path))

    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def load_runner_metrics(metrics_file: str) -> dict[str, Any]:
    """Load runner metrics JSON from disk."""
    path = Path(metrics_file)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def build_validation_result_from_artifacts(
    *,
    config: PaperTradingConfig,
    metrics_file: str,
    events_file: str,
    run_id: str | None = None,
) -> ValidationResult:
    """Build a validation result from strategy runner artifacts.

    This converts the existing Phase 2-4 runner metrics/events into a Phase 5
    validation result without requiring live broker connectivity.
    """
    metrics = load_runner_metrics(metrics_file)
    events = load_jsonl_events(events_file)

    trade_closed_events = [event for event in events if event.get("kind") == "trade_closed"]
    rejection_events = [event for event in events if event.get("kind") in {"rejection_burst_detected", "order_submit"}]
    de_risk_events = [event for event in events if event.get("kind") == "de_risk_enabled"]
    kill_switch_events = [event for event in events if str(event.get("kind", "")).startswith("kill_switch")]
    state_transition_events = [event for event in events if event.get("kind") == "state_transition"]

    pnls = [float(event.get("payload", {}).get("pnl_pct", 0.0)) for event in trade_closed_events]
    winning_trades = sum(1 for pnl in pnls if pnl > 0)
    losing_trades = sum(1 for pnl in pnls if pnl < 0)
    total_trades = len(pnls)

    consecutive_losses = 0
    max_consecutive_losses = 0
    equity = float(config.base_capital)
    peak_equity = equity
    max_drawdown_pct = 0.0
    max_daily_loss_pct = 0.0

    for pnl_pct in pnls:
        if pnl_pct < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

        equity *= 1.0 + pnl_pct
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            drawdown_pct = max(0.0, (peak_equity - equity) / peak_equity * 100.0)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        daily_loss_pct = max(0.0, (config.base_capital - equity) / config.base_capital * 100.0)
        max_daily_loss_pct = max(max_daily_loss_pct, daily_loss_pct)

    gross_pnl = equity - float(config.base_capital)
    gross_pnl_pct = 0.0 if config.base_capital <= 0 else gross_pnl / float(config.base_capital) * 100.0
    win_rate_pct = 0.0 if total_trades == 0 else winning_trades / total_trades * 100.0

    latency = metrics.get("latency_ms", {}).get("loop", {})
    start_time = events[0].get("ts") if events else datetime.now(timezone.utc).isoformat()
    end_time = events[-1].get("ts") if events else start_time
    duration_seconds = max(0.0, _parse_iso_timestamp(end_time) - _parse_iso_timestamp(start_time))

    result = ValidationResult(
        run_id=run_id or f"papertrading_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        start_time=str(start_time),
        end_time=str(end_time),
        duration_seconds=float(duration_seconds),
        symbol=str(metrics.get("symbol", config.symbol)),
        broker=str(config.broker),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        max_consecutive_losses_observed=max_consecutive_losses,
        win_rate_pct=float(win_rate_pct),
        starting_cash=float(config.base_capital),
        final_cash=float(equity),
        gross_pnl=float(gross_pnl),
        gross_pnl_pct=float(gross_pnl_pct),
        best_trade_pct=max(pnls) * 100.0 if pnls else 0.0,
        worst_trade_pct=min(pnls) * 100.0 if pnls else 0.0,
        max_daily_loss_pct=float(max_daily_loss_pct),
        max_rolling_drawdown_pct=float(max_drawdown_pct),
        peak_equity=float(peak_equity),
        final_equity=float(equity),
        rejections_encountered=sum(
            1
            for event in rejection_events
            if str(event.get("payload", {}).get("status", "")).lower() in {"rejected", "reject"}
            or event.get("kind") == "rejection_burst_detected"
        ),
        de_risk_mode_activations=len(de_risk_events),
        kill_switch_triggers=len(kill_switch_events)
        + sum(
            1
            for event in state_transition_events
            if str(event.get("payload", {}).get("state", "")).upper() in {"RISK_LOCK", "HALTED"}
        ),
        passed_validation=False,
        validation_errors=[],
        recommendations=[],
        p95_latency_ms=float(latency.get("p95", 0.0) or 0.0),
        p99_latency_ms=float(latency.get("p99", latency.get("p95", 0.0)) or 0.0),
        avg_latency_ms=float(latency.get("mean", 0.0) or 0.0),
    )
    return result


def save_validation_result_json(result: ValidationResult, output_file: str) -> None:
    """Persist validation result as JSON."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def load_validation_result_json(file_path: str) -> ValidationResult:
    """Load ValidationResult from JSON file."""
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return ValidationResult(**data)


def calibrate_promotion_gates(
    validation_files: list[str],
    *,
    min_reports_required: int = 3,
) -> PromotionGateCalibration:
    """Calibrate promotion gates based on historical paper-trading reports."""
    reports = [load_validation_result_json(path) for path in validation_files]
    num_reports = len(reports)
    notes: list[str] = []

    if num_reports == 0:
        return PromotionGateCalibration(
            num_reports=0,
            min_reports_required=min_reports_required,
            ready_for_enforcement=False,
            source_files=[],
            recommended_min_trades=20,
            recommended_min_win_rate_pct=55.0,
            recommended_max_consecutive_loss_streak=3,
            recommended_max_rolling_drawdown_pct=5.0,
            recommended_max_p95_latency_ms=1000.0,
            notes=["No reports found. Using conservative defaults."],
        )

    total_trades = [max(0, int(r.total_trades)) for r in reports]
    win_rates = [max(0.0, float(r.win_rate_pct)) for r in reports]
    loss_streaks = [max(0, int(r.max_consecutive_losses_observed)) for r in reports]
    drawdowns = [max(0.0, float(r.max_rolling_drawdown_pct)) for r in reports]
    p95_latencies = [max(0.0, float(r.p95_latency_ms)) for r in reports]

    # Percentile-style robust calibration without external deps.
    def _pct(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        values_sorted = sorted(values)
        idx = min(len(values_sorted) - 1, max(0, int(round((pct / 100.0) * (len(values_sorted) - 1)))))
        return float(values_sorted[idx])

    recommended_min_trades = max(5, int(_pct([float(v) for v in total_trades], 40)))
    recommended_min_win_rate_pct = max(50.0, min(75.0, _pct(win_rates, 35)))
    recommended_max_consecutive_loss_streak = max(2, int(_pct([float(v) for v in loss_streaks], 80)) + 1)
    recommended_max_rolling_drawdown_pct = max(2.0, min(10.0, _pct(drawdowns, 80) + 0.5))
    recommended_max_p95_latency_ms = max(200.0, min(3000.0, _pct(p95_latencies, 85) + 50.0))

    ready = num_reports >= int(min_reports_required)
    if not ready:
        notes.append(
            f"Collected {num_reports} report(s); need at least {min_reports_required} before hard enforcement."
        )
    if all(int(r.total_trades) == 0 for r in reports):
        notes.append("All reports have zero closed trades; treat calibrated win-rate gate as provisional.")

    return PromotionGateCalibration(
        num_reports=num_reports,
        min_reports_required=min_reports_required,
        ready_for_enforcement=ready,
        source_files=[str(path) for path in validation_files],
        recommended_min_trades=recommended_min_trades,
        recommended_min_win_rate_pct=recommended_min_win_rate_pct,
        recommended_max_consecutive_loss_streak=recommended_max_consecutive_loss_streak,
        recommended_max_rolling_drawdown_pct=recommended_max_rolling_drawdown_pct,
        recommended_max_p95_latency_ms=recommended_max_p95_latency_ms,
        notes=notes,
    )


def save_promotion_gate_calibration(calibration: PromotionGateCalibration, output_file: str) -> None:
    """Persist promotion gate calibration as JSON."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")


def build_phase6_tuning_plan(
    validation_files: list[str],
    *,
    default_poll_seconds: float = 0.2,
) -> Phase6TuningPlan:
    """Build a Phase 6 performance tuning plan from validation reports."""
    reports = [load_validation_result_json(path) for path in validation_files]
    if not reports:
        return Phase6TuningPlan(
            num_reports=0,
            profile_counts={},
            recommended_execution_profile="scalper",
            recommended_poll_seconds=default_poll_seconds,
            recommended_max_iterations=10,
            suggested_hard_daily_loss_pct=2.0,
            suggested_rolling_drawdown_pct=5.0,
            suggested_max_consecutive_losses=4,
            priority_actions=["Collect paper-trading reports before tuning."],
        )

    profile_counts: dict[str, int] = {}
    for report in reports:
        # Best available proxy: encode via run id naming convention if present, else unknown.
        run_id = str(report.run_id).lower()
        profile = "scalper"
        for candidate in ("conservative", "moderate", "scalper"):
            if candidate in run_id:
                profile = candidate
                break
        profile_counts[profile] = profile_counts.get(profile, 0) + 1

    p95s = [float(r.p95_latency_ms) for r in reports if float(r.p95_latency_ms) > 0]
    avg_p95 = sum(p95s) / len(p95s) if p95s else 500.0
    recommended_poll_seconds = max(0.2, min(2.0, avg_p95 / 1000.0))

    drawdowns = [float(r.max_rolling_drawdown_pct) for r in reports]
    avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0
    suggested_rolling_dd = max(3.0, min(6.0, avg_drawdown + 1.0))

    # Prefer a more conservative profile if drawdowns are elevated.
    if avg_drawdown > 4.0:
        recommended_profile = "conservative"
    elif avg_drawdown > 2.5:
        recommended_profile = "moderate"
    else:
        recommended_profile = "scalper"

    priority_actions = [
        "Tune poll interval to keep p95 loop latency stable under 1s.",
        "Run side-by-side profile comparison for scalper/moderate/conservative.",
        "Use calibrated promotion gates before enabling canary deployment.",
    ]

    return Phase6TuningPlan(
        num_reports=len(reports),
        profile_counts=profile_counts,
        recommended_execution_profile=recommended_profile,
        recommended_poll_seconds=float(round(recommended_poll_seconds, 3)),
        recommended_max_iterations=max(10, len(reports) * 5),
        suggested_hard_daily_loss_pct=2.0,
        suggested_rolling_drawdown_pct=float(round(suggested_rolling_dd, 2)),
        suggested_max_consecutive_losses=4,
        priority_actions=priority_actions,
    )


def save_phase6_tuning_plan(plan: Phase6TuningPlan, output_file: str) -> None:
    """Persist Phase 6 tuning plan as JSON."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")


def _parse_iso_timestamp(value: str) -> float:
    """Parse ISO timestamp into unix seconds."""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except Exception:
        return datetime.now(timezone.utc).timestamp()


def create_paper_trading_validation_report(
    result: ValidationResult,
    output_file: str,
) -> None:
    """Create HTML validation report from paper trading results.
    
    Args:
        result: ValidationResult from paper trading run
        output_file: Path to write HTML report
    """
    status_color = "#00aa00" if result.passed_validation else "#ff6600"
    status_text = "✓ PASSED" if result.passed_validation else "⚠ REVIEW NEEDED"
    
    html = f"""
    <html>
    <head>
        <title>Paper Trading Validation Report - {result.run_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }}
            .header {{ background: {status_color}; color: white; padding: 20px; border-radius: 8px; }}
            .section {{ background: white; margin: 10px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; background: #f0f0f0; padding: 10px; margin: 5px; border-radius: 4px; }}
            .good {{ background: #e8f5e9; color: #2e7d32; }}
            .warning {{ background: #fff3e0; color: #c67c06; }}
            .error {{ background: #ffebee; color: #c62828; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #f0f0f0; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Paper Trading Validation Report</h1>
            <p><strong>Status: {status_text}</strong></p>
            <p>Run ID: {result.run_id}</p>
            <p>Duration: {result.duration_seconds / 3600:.1f} hours ({result.start_time} to {result.end_time})</p>
        </div>
        
        <div class="section">
            <h2>Trading Summary</h2>
            <div class="metric">Total Trades: <strong>{result.total_trades}</strong></div>
            <div class="metric good">Wins: {result.winning_trades}</div>
            <div class="metric warning">Losses: {result.losing_trades}</div>
            <div class="metric">Win Rate: <strong>{result.win_rate_pct:.1f}%</strong></div>
        </div>
        
        <div class="section">
            <h2>Financial Results</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Starting Capital</td>
                    <td>${result.starting_cash:.2f}</td>
                </tr>
                <tr>
                    <td>Final Capital</td>
                    <td>${result.final_cash:.2f}</td>
                </tr>
                <tr>
                    <td>Gross PnL</td>
                    <td>${result.gross_pnl:.2f} ({result.gross_pnl_pct:+.2f}%)</td>
                </tr>
                <tr>
                    <td>Best Trade</td>
                    <td>{result.best_trade_pct:+.2f}%</td>
                </tr>
                <tr>
                    <td>Worst Trade</td>
                    <td>{result.worst_trade_pct:+.2f}%</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Risk Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Peak Equity</td>
                    <td>${result.peak_equity:.2f}</td>
                </tr>
                <tr>
                    <td>Max Daily Loss</td>
                    <td>{result.max_daily_loss_pct:.2f}%</td>
                </tr>
                <tr>
                    <td>Max Rolling Drawdown</td>
                    <td>{result.max_rolling_drawdown_pct:.2f}%</td>
                </tr>
                <tr>
                    <td>Kill Switch Triggers</td>
                    <td>{result.kill_switch_triggers}</td>
                </tr>
                <tr>
                    <td>De-risk Activations</td>
                    <td>{result.de_risk_mode_activations}</td>
                </tr>
                <tr>
                    <td>Max Consecutive Losses Observed</td>
                    <td>{result.max_consecutive_losses_observed}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Latency Performance</h2>
            <div class="metric">P95: {result.p95_latency_ms:.0f}ms</div>
            <div class="metric">P99: {result.p99_latency_ms:.0f}ms</div>
            <div class="metric">Average: {result.avg_latency_ms:.0f}ms</div>
        </div>
    """
    
    if result.validation_errors:
        html += """
        <div class="section error">
            <h2>⚠ Validation Issues</h2>
            <ul>
        """
        for error in result.validation_errors:
            html += f"<li>{error}</li>"
        html += "</ul></div>"
    
    if result.recommendations:
        html += """
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
        """
        for rec in result.recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul></div>"
    
    html += """
    </body>
    </html>
    """
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(html)
