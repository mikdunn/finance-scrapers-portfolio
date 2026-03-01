"""Master-brain governance pass for research outputs.

Goals:
- Audit backtest runs against practical finance/risk principles.
- Audit ML outputs for robustness signals (walk-forward stability when available).
- Produce a safe, explicit folder cleanup plan for output artifacts.
- Optionally apply folder moves into a cleaner taxonomy.

This script is intentionally conservative: default behavior is dry-run planning.

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    min_trades: int = 30
    min_sharpe: float = 0.5
    max_drawdown_abs: float = 0.35
    min_profit_factor: float = 1.05
    min_cv_folds: int = 4
    min_reliability_score: float = 0.60
    min_ml_quality_score: float = 0.55


def _safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_run_from_model_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    try:
        p = Path(str(model_path))
        return p.parent.name if p.parent is not None else None
    except Exception:
        return None


def _parse_bps_grid(s: str | None, *, default: str = "10,20,35") -> list[float]:
    raw = default if s is None or not str(s).strip() else str(s)
    vals: list[float] = []
    for t in re.split(r"[\s,;]+", raw.strip()):
        if not t:
            continue
        try:
            x = float(t)
            if x >= 0:
                vals.append(x)
        except Exception:
            continue
    if not vals:
        vals = [10.0, 20.0, 35.0]
    vals = sorted(set(vals))
    return vals


def _calibration_quality_from_metrics(m: dict) -> tuple[float, dict]:
    """Map calibration diagnostics into a bounded quality score [0,1]."""
    if not isinstance(m, dict):
        return 0.0, {"cal_present": False}

    ece = _safe_float(m.get("cal_ece"))
    brier = _safe_float(m.get("cal_brier_multi"))
    logloss = _safe_float(m.get("cal_log_loss"))

    if not any(np.isfinite(x) for x in (ece, brier, logloss)):
        return 0.0, {"cal_present": False}

    # Lower is better for all three metrics.
    ece_term = float(np.clip(1.0 - (ece / 0.25), 0.0, 1.0)) if np.isfinite(ece) else float("nan")
    brier_term = float(np.clip(1.0 - (brier / 0.75), 0.0, 1.0)) if np.isfinite(brier) else float("nan")
    ll_term = float(1.0 / (1.0 + max(logloss, 0.0))) if np.isfinite(logloss) else float("nan")

    terms = [x for x in (ece_term, brier_term, ll_term) if np.isfinite(x)]
    if not terms:
        return 0.0, {"cal_present": False}

    score = float(np.mean(terms))
    return score, {
        "cal_present": True,
        "cal_ece": ece,
        "cal_brier_multi": brier,
        "cal_log_loss": logloss,
    }


def _discover_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _classify_dir(name: str) -> str:
    n = name.lower()
    if n.startswith("bt_") or n.startswith("backtest_"):
        return "backtests"
    if n.startswith("backtest_report"):
        return "reports"
    if n.startswith("ml_outputs"):
        return "ml"
    if n.startswith("hub_"):
        return "hub"
    if n.startswith("outputs"):
        return "outputs"
    if n.startswith("smoke_"):
        return "smoke"
    return "other"


def _target_for_category(cat: str) -> Path:
    mapping = {
        "backtests": Path("artifacts/backtests"),
        "reports": Path("artifacts/reports"),
        "ml": Path("artifacts/ml"),
        "hub": Path("artifacts/hub"),
        "outputs": Path("artifacts/outputs"),
        "smoke": Path("artifacts/smoke"),
    }
    return mapping.get(cat, Path("artifacts/misc"))


def _annualized_score(sharpe: float, drawdown: float, cagr: float) -> float:
    # Smooth bounded transforms for robust ranking.
    s = float(np.tanh(np.nan_to_num(sharpe, nan=0.0) / 2.0))
    dd_abs = abs(drawdown) if np.isfinite(drawdown) else 1.0
    d = float(np.clip(1.0 - dd_abs / 0.6, 0.0, 1.0))
    g = float(np.tanh(np.nan_to_num(cagr, nan=0.0) / 0.35))
    # Emphasize risk-adjusted performance over raw growth.
    return 0.50 * s + 0.35 * d + 0.15 * g


def _trade_quality_score(win_rate: float, profit_factor: float, n_trades: int, min_trades: int) -> float:
    wr = np.nan_to_num(win_rate, nan=0.0)
    pf = np.nan_to_num(profit_factor, nan=0.0)
    wr_term = float(np.clip((wr - 0.45) / 0.20, 0.0, 1.0))
    pf_term = float(np.clip((pf - 1.0) / 1.0, 0.0, 1.0))
    sample_term = float(np.clip(n_trades / max(1.0, float(min_trades)), 0.0, 1.0))
    return 0.35 * wr_term + 0.45 * pf_term + 0.20 * sample_term


def _cv_quality_score(ml_dir: Path, policy: Policy) -> tuple[float, dict]:
    cv_csv = ml_dir / "cv_metrics.csv"
    if not cv_csv.exists():
        return 0.0, {"cv_present": False}

    try:
        cv = pd.read_csv(cv_csv)
    except Exception:
        return 0.0, {"cv_present": False, "cv_error": "read_failed"}

    if cv.empty:
        return 0.0, {"cv_present": False, "cv_error": "empty"}

    out = {"cv_present": True, "n_folds": int(len(cv))}
    n_folds = int(len(cv))
    fold_term = float(np.clip(n_folds / max(1.0, float(policy.min_cv_folds)), 0.0, 1.0))

    if "f1_macro" in cv.columns:
        vals = pd.to_numeric(cv["f1_macro"], errors="coerce").dropna()
        if vals.empty:
            return 0.0, {**out, "cv_metric": "f1_macro", "cv_error": "nan"}
        mean = float(vals.mean())
        std = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        stability = float(np.clip(1.0 - std / 0.20, 0.0, 1.0))
        quality = float(np.clip((mean - 0.33) / 0.40, 0.0, 1.0))
        score = 0.35 * fold_term + 0.45 * quality + 0.20 * stability
        return score, {**out, "cv_metric": "f1_macro", "cv_mean": mean, "cv_std": std}

    if "rmse" in cv.columns:
        vals = pd.to_numeric(cv["rmse"], errors="coerce").dropna()
        if vals.empty:
            return 0.0, {**out, "cv_metric": "rmse", "cv_error": "nan"}
        mean = float(vals.mean())
        std = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        # Lower RMSE is better; normalize with a soft inverse map.
        quality = float(1.0 / (1.0 + max(mean, 0.0)))
        stability = float(np.clip(1.0 - std / max(1e-9, mean + 1e-9), 0.0, 1.0))
        score = 0.35 * fold_term + 0.45 * quality + 0.20 * stability
        return score, {**out, "cv_metric": "rmse", "cv_mean": mean, "cv_std": std}

    return 0.0, {**out, "cv_metric": "unknown"}


def _load_backtest_row(bt_dir: Path, policy: Policy) -> dict | None:
    summary_path = bt_dir / "backtest_summary.json"
    if not summary_path.exists():
        return None

    try:
        s = _load_json(summary_path)
    except Exception:
        return None

    metrics = s.get("portfolio_metrics") if isinstance(s.get("portfolio_metrics"), dict) else s.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    trade_metrics = s.get("trade_metrics") if isinstance(s.get("trade_metrics"), dict) else {}

    sharpe = _safe_float(metrics.get("sharpe"))
    cagr = _safe_float(metrics.get("cagr"))
    dd = _safe_float(metrics.get("max_drawdown"))
    total_ret = _safe_float(metrics.get("total_return"))
    n_bars = _safe_float(metrics.get("n_bars"))
    turnover_sum = _safe_float(metrics.get("turnover_sum"))
    avg_bar_return = _safe_float(metrics.get("avg_bar_return"))
    vol_bar = _safe_float(metrics.get("vol_bar"))
    ann_factor = _safe_float(metrics.get("ann_factor"))

    win_rate = _safe_float(trade_metrics.get("win_rate"))
    profit_factor = _safe_float(trade_metrics.get("profit_factor"))
    n_trades = int(_safe_float(trade_metrics.get("n_trades")) if np.isfinite(_safe_float(trade_metrics.get("n_trades"))) else 0)

    finance_score = _annualized_score(sharpe=sharpe, drawdown=dd, cagr=cagr)
    trade_score = _trade_quality_score(
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=n_trades,
        min_trades=policy.min_trades,
    )

    # Reliability penalties for principle violations.
    principles: list[str] = []
    if np.isfinite(sharpe) and sharpe < policy.min_sharpe:
        principles.append("low_sharpe")
    if np.isfinite(dd) and abs(dd) > policy.max_drawdown_abs:
        principles.append("drawdown_too_high")
    if n_trades > 0 and n_trades < policy.min_trades:
        principles.append("too_few_trades")
    if np.isfinite(profit_factor) and profit_factor < policy.min_profit_factor:
        principles.append("weak_profit_factor")

    penalty = 0.08 * len(principles)
    penalty = float(np.clip(penalty, 0.0, 0.35))

    reliability = float(np.clip(0.65 * finance_score + 0.35 * trade_score - penalty, 0.0, 1.0))

    return {
        "run": bt_dir.name,
        "path": str(bt_dir),
        "asset": s.get("asset"),
        "model": s.get("model"),
        "model_run": _model_run_from_model_path(s.get("model")),
        "portfolio_weighting": s.get("portfolio_weighting"),
        "signal_source": s.get("signal_source"),
        "mode": s.get("mode"),
        "m_total_return": total_ret,
        "m_cagr": cagr,
        "m_sharpe": sharpe,
        "m_max_drawdown": dd,
        "m_n_bars": n_bars,
        "m_turnover_sum": turnover_sum,
        "m_avg_bar_return": avg_bar_return,
        "m_vol_bar": vol_bar,
        "m_ann_factor": ann_factor,
        "t_n_trades": n_trades,
        "t_win_rate": win_rate,
        "t_profit_factor": profit_factor,
        "finance_score": finance_score,
        "trade_score": trade_score,
        "reliability_score": reliability,
        "principle_flags": ",".join(principles),
        "n_principle_flags": int(len(principles)),
    }


def _scan_backtests(root: Path, policy: Policy) -> pd.DataFrame:
    rows: list[dict] = []
    for d in _discover_dirs(root):
        if _classify_dir(d.name) != "backtests":
            continue
        row = _load_backtest_row(d, policy=policy)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["reliability_score", "m_sharpe", "m_total_return"], ascending=False)
    return df.reset_index(drop=True)


def _scan_ml(root: Path, policy: Policy) -> pd.DataFrame:
    rows: list[dict] = []
    for d in _discover_dirs(root):
        if _classify_dir(d.name) != "ml":
            continue

        metrics_path = d / "metrics.json"
        m = {}
        if metrics_path.exists():
            try:
                m = _load_json(metrics_path)
            except Exception:
                m = {}

        cv_score, cv_meta = _cv_quality_score(d, policy=policy)
        cal_score, cal_meta = _calibration_quality_from_metrics(m)

        # Prefer a blend when calibration metrics exist.
        if bool(cal_meta.get("cal_present")):
            ml_quality = float(np.clip(0.55 * cv_score + 0.45 * cal_score, 0.0, 1.0))
        else:
            ml_quality = float(np.clip(cv_score, 0.0, 1.0))

        row = {
            "run": d.name,
            "path": str(d),
            "task": m.get("task"),
            "model": m.get("model"),
            "mode": m.get("mode"),
            "feature_count": _safe_float(m.get("feature_count")),
            "cv_quality_score": float(np.clip(cv_score, 0.0, 1.0)),
            "calibration_quality_score": float(np.clip(cal_score, 0.0, 1.0)),
            "ml_quality_score": ml_quality,
            **cv_meta,
            **cal_meta,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["ml_quality_score", "run"], ascending=[False, True]).reset_index(drop=True)


def _apply_cost_stress(bt_df: pd.DataFrame, *, stress_bps_grid: list[float]) -> pd.DataFrame:
    if bt_df is None or bt_df.empty:
        return bt_df

    out = bt_df.copy()

    for bps in stress_bps_grid:
        b = int(round(float(bps)))
        rate = float(bps) / 10000.0

        trn = pd.to_numeric(out.get("m_turnover_sum"), errors="coerce")
        n_bars = pd.to_numeric(out.get("m_n_bars"), errors="coerce")
        avg = pd.to_numeric(out.get("m_avg_bar_return"), errors="coerce")
        vol = pd.to_numeric(out.get("m_vol_bar"), errors="coerce")
        ann = pd.to_numeric(out.get("m_ann_factor"), errors="coerce")
        total = pd.to_numeric(out.get("m_total_return"), errors="coerce")

        extra_drag = rate * trn
        out[f"stress_total_return_bps{b}"] = total - extra_drag

        per_bar_drag = np.where((n_bars > 0) & np.isfinite(n_bars), extra_drag / n_bars, np.nan)
        avg_new = avg - per_bar_drag
        sharpe_new = np.where(
            (vol > 0) & np.isfinite(vol) & (ann > 0) & np.isfinite(ann),
            (avg_new / vol) * np.sqrt(ann),
            np.nan,
        )
        out[f"stress_sharpe_bps{b}"] = sharpe_new

    return out


def _build_promotion_gate(
    bt_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    *,
    policy: Policy,
    gate_stress_bps: float,
) -> pd.DataFrame:
    if bt_df is None or bt_df.empty:
        return pd.DataFrame()

    out = bt_df.copy()
    out["model_run"] = out.get("model_run")

    if ml_df is not None and not ml_df.empty and "run" in ml_df.columns:
        keep = [
            c
            for c in [
                "run",
                "ml_quality_score",
                "cv_present",
                "n_folds",
                "cv_metric",
                "cv_mean",
                "cv_std",
                "calibration_quality_score",
                "cal_ece",
                "cal_brier_multi",
                "cal_log_loss",
            ]
            if c in ml_df.columns
        ]
        m = ml_df[keep].rename(columns={"run": "model_run"})
        out = out.merge(m, on="model_run", how="left")

    sb = int(round(float(gate_stress_bps)))
    sret_col = f"stress_total_return_bps{sb}"
    ssh_col = f"stress_sharpe_bps{sb}"

    sret = pd.to_numeric(out.get(sret_col), errors="coerce") if sret_col in out.columns else pd.Series(np.nan, index=out.index)
    ssh = pd.to_numeric(out.get(ssh_col), errors="coerce") if ssh_col in out.columns else pd.Series(np.nan, index=out.index)
    mlq = pd.to_numeric(out.get("ml_quality_score"), errors="coerce") if "ml_quality_score" in out.columns else pd.Series(np.nan, index=out.index)
    rel = pd.to_numeric(out.get("reliability_score"), errors="coerce")
    n_flags = pd.to_numeric(out.get("n_principle_flags"), errors="coerce").fillna(0)
    cv_present = out.get("cv_present", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    out["pass_finance"] = (rel >= float(policy.min_reliability_score)) & (n_flags <= 0)
    out["pass_stress"] = (sret > 0.0) & (ssh >= float(policy.min_sharpe))
    out["pass_ml"] = (mlq >= float(policy.min_ml_quality_score)) & cv_present

    out["promotion_score"] = (
        0.55 * np.nan_to_num(rel, nan=0.0)
        + 0.30 * np.nan_to_num(mlq, nan=0.0)
        + 0.15 * np.clip(np.nan_to_num((ssh / 2.0), nan=0.0), 0.0, 1.0)
    )

    decision = np.where(
        out["pass_finance"] & out["pass_stress"] & out["pass_ml"],
        "promote",
        np.where(out["pass_finance"] & out["pass_stress"], "candidate", "reject"),
    )
    out["decision"] = decision

    cols_front = [
        "run",
        "model_run",
        "decision",
        "promotion_score",
        "pass_finance",
        "pass_stress",
        "pass_ml",
        "reliability_score",
        "ml_quality_score",
    ]
    ordered = [c for c in cols_front if c in out.columns] + [c for c in out.columns if c not in cols_front]
    out = out[ordered]
    return out.sort_values(["decision", "promotion_score"], ascending=[True, False]).reset_index(drop=True)


def _write_phase2_dashboard(
    bt_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    *,
    stress_grid: list[float],
    gate_stress_bps: float,
    out_html: Path,
) -> bool:
    """Write an interactive governance dashboard (best-effort)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    if gate_df is None or gate_df.empty:
        out_html.write_text("<html><body><h2>No promotion-gate rows available.</h2></body></html>", encoding="utf-8")
        return True

    g = gate_df.copy()

    # Ensure required numeric fields are usable.
    g["promotion_score"] = pd.to_numeric(g.get("promotion_score"), errors="coerce")
    g["reliability_score"] = pd.to_numeric(g.get("reliability_score"), errors="coerce")
    g["ml_quality_score"] = pd.to_numeric(g.get("ml_quality_score"), errors="coerce")

    decision_order = ["promote", "candidate", "reject"]
    decision_colors = {
        "promote": "#2ca02c",
        "candidate": "#ff7f0e",
        "reject": "#d62728",
    }

    # Stress panel: top runs by promotion_score, show stressed Sharpe across grid.
    stress_cols: list[tuple[float, str]] = []
    for b in stress_grid:
        c = f"stress_sharpe_bps{int(round(float(b)))}"
        if c in g.columns:
            stress_cols.append((float(b), c))
    stress_cols = sorted(stress_cols, key=lambda x: x[0])

    top = g.sort_values("promotion_score", ascending=False).head(8).copy()

    # Build dashboard layout.
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Promotion decisions",
            "Finance vs ML quality",
            "Stress Sharpe trajectories (top runs)",
            "Calibration diagnostics",
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    # (1) Decision counts
    counts = g["decision"].astype(str).value_counts()
    x = decision_order
    y = [int(counts.get(k, 0)) for k in x]
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            marker_color=[decision_colors.get(k, "#888") for k in x],
            name="decision_count",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # (2) Finance vs ML scatter
    for d in decision_order:
        dd = g[g["decision"].astype(str) == d].copy()
        if dd.empty:
            continue
        size = np.clip(np.nan_to_num(dd["promotion_score"].to_numpy(), nan=0.0), 0.0, 1.0)
        size = 10 + 20 * size
        fig.add_trace(
            go.Scatter(
                x=dd["reliability_score"],
                y=dd["ml_quality_score"],
                mode="markers+text",
                text=dd["run"],
                textposition="top center",
                marker=dict(size=size, color=decision_colors.get(d, "#888"), opacity=0.85),
                name=d,
                legendgroup=d,
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # (3) Stress Sharpe trajectories
    if stress_cols and not top.empty:
        xs = [b for b, _ in stress_cols]
        for _, r in top.iterrows():
            ys: list[float] = []
            for _, c in stress_cols:
                ys.append(_safe_float(r.get(c)))
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    name=str(r.get("run")),
                    showlegend=False,
                    hovertemplate=f"run={r.get('run')}<br>stress_bps=%{{x}}<br>stress_sharpe=%{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # Marker line at gate stress bps on stress subplot.
    if stress_cols:
        fig.add_vline(
            x=float(gate_stress_bps),
            line_width=1,
            line_dash="dash",
            line_color="#666",
            row=2,
            col=1,
        )

    # (4) Calibration diagnostics from ML table (if available)
    cal_ok = False
    if ml_df is not None and not ml_df.empty and "cal_ece" in ml_df.columns and "ml_quality_score" in ml_df.columns:
        m = ml_df.copy()
        m["cal_ece"] = pd.to_numeric(m.get("cal_ece"), errors="coerce")
        m["ml_quality_score"] = pd.to_numeric(m.get("ml_quality_score"), errors="coerce")
        m = m.dropna(subset=["cal_ece", "ml_quality_score"])
        if not m.empty:
            cal_ok = True
            fig.add_trace(
                go.Scatter(
                    x=m["cal_ece"],
                    y=m["ml_quality_score"],
                    mode="markers+text",
                    text=m["run"],
                    textposition="top center",
                    marker=dict(size=10, color="#1f77b4", opacity=0.85),
                    name="ml_calibration",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    # Axis labels
    fig.update_xaxes(title_text="Decision", row=1, col=1)
    fig.update_yaxes(title_text="# Runs", row=1, col=1)

    fig.update_xaxes(title_text="Reliability score", row=1, col=2)
    fig.update_yaxes(title_text="ML quality score", row=1, col=2)

    fig.update_xaxes(title_text="Stress cost (bps)", row=2, col=1)
    fig.update_yaxes(title_text="Stressed Sharpe", row=2, col=1)

    fig.update_xaxes(title_text="Calibration ECE (lower is better)", row=2, col=2)
    fig.update_yaxes(title_text="ML quality score", row=2, col=2)

    title_suffix = ""
    if not cal_ok:
        title_suffix = " (calibration panel unavailable: no cal_ece data)"

    fig.update_layout(
        template="plotly_white",
        height=980,
        title=f"Master Brain Governance Dashboard{title_suffix}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
    return True


def _write_promotion_memo(
    gate_df: pd.DataFrame,
    *,
    out_md: Path,
    gate_stress_bps: float,
    top_n: int = 10,
) -> None:
    """Write an investment-committee style Top-N promotion memo."""
    lines: list[str] = []
    lines.append("# Top Promotion Memo")
    lines.append("")
    lines.append("This memo is auto-generated by `master_brain` and summarizes the highest-ranked strategy runs for deployment review.")
    lines.append("")

    if gate_df is None or gate_df.empty:
        lines.append("## Summary")
        lines.append("")
        lines.append("- No promotion-gate rows were available.")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    g = gate_df.copy()
    g["promotion_score"] = pd.to_numeric(g.get("promotion_score"), errors="coerce")
    g["reliability_score"] = pd.to_numeric(g.get("reliability_score"), errors="coerce")
    g["ml_quality_score"] = pd.to_numeric(g.get("ml_quality_score"), errors="coerce")

    sb = int(round(float(gate_stress_bps)))
    sret_col = f"stress_total_return_bps{sb}"
    ssh_col = f"stress_sharpe_bps{sb}"

    if sret_col in g.columns:
        g[sret_col] = pd.to_numeric(g[sret_col], errors="coerce")
    if ssh_col in g.columns:
        g[ssh_col] = pd.to_numeric(g[ssh_col], errors="coerce")

    rank = g.sort_values(["promotion_score", "reliability_score"], ascending=False).head(int(max(1, top_n))).reset_index(drop=True)
    promoted = g[g.get("decision").astype(str) == "promote"].copy() if "decision" in g.columns else pd.DataFrame()

    def _light_from_bool(ok: bool, *, amber_if: bool = False) -> str:
        if bool(ok):
            return "🟢"
        if bool(amber_if):
            return "🟠"
        return "🔴"

    def _overall_light(decision: str) -> str:
        d = str(decision).strip().lower()
        if d == "promote":
            return "🟢"
        if d == "candidate":
            return "🟠"
        return "🔴"

    counts = g["decision"].astype(str).value_counts()
    lines.append("## Decision distribution")
    lines.append("")
    lines.append(f"- promote: **{int(counts.get('promote', 0))}**")
    lines.append(f"- candidate: **{int(counts.get('candidate', 0))}**")
    lines.append(f"- reject: **{int(counts.get('reject', 0))}**")
    lines.append("")

    lines.append("## Traffic-light go/no-go table")
    lines.append("")
    lines.append("Legend: 🟢 pass, 🟠 monitor/conditional, 🔴 fail")
    lines.append("")
    lines.append("| Run | Decision | Overall | Finance | ML | Stress | Ops |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for _, r in rank.iterrows():
        run = str(r.get("run"))
        decision = str(r.get("decision") or "reject")

        rel = _safe_float(r.get("reliability_score"))
        mlq = _safe_float(r.get("ml_quality_score"))
        pass_finance = bool(r.get("pass_finance"))
        pass_ml = bool(r.get("pass_ml"))
        pass_stress = bool(r.get("pass_stress"))
        cv_present = bool(r.get("cv_present")) if "cv_present" in r.index else False
        sret = _safe_float(r.get(sret_col)) if sret_col in rank.columns else float("nan")
        ssh = _safe_float(r.get(ssh_col)) if ssh_col in rank.columns else float("nan")
        cal_ece = _safe_float(r.get("cal_ece"))
        flags = str(r.get("principle_flags") or "none")

        finance_light = _light_from_bool(pass_finance, amber_if=(np.isfinite(rel) and rel >= 0.50))
        ml_light = _light_from_bool(pass_ml, amber_if=(cv_present and np.isfinite(mlq) and mlq >= 0.45))
        stress_light = _light_from_bool(
            pass_stress,
            amber_if=(np.isfinite(sret) and np.isfinite(ssh) and sret > -0.05 and ssh >= 0.40),
        )

        # Operational light: fail if principle flags exist; amber if calibration missing/weak.
        ops_ok = flags.strip().lower() in {"", "none"}
        ops_amber = (not np.isfinite(cal_ece)) or (np.isfinite(cal_ece) and cal_ece <= 0.20)
        ops_light = _light_from_bool(ops_ok and np.isfinite(cal_ece) and cal_ece <= 0.12, amber_if=ops_ok and ops_amber)

        lines.append(
            f"| `{run}` | `{decision}` | {_overall_light(decision)} | {finance_light} | {ml_light} | {stress_light} | {ops_light} |"
        )

    lines.append("")

    lines.append(f"## Top {len(rank)} ranked runs")
    lines.append("")

    for i, r in rank.iterrows():
        run = str(r.get("run"))
        decision = str(r.get("decision"))
        score = _safe_float(r.get("promotion_score"))
        rel = _safe_float(r.get("reliability_score"))
        mlq = _safe_float(r.get("ml_quality_score"))
        sharpe = _safe_float(r.get("m_sharpe"))
        dd = _safe_float(r.get("m_max_drawdown"))
        ntr = _safe_float(r.get("t_n_trades"))
        pf = _safe_float(r.get("t_profit_factor"))
        sret = _safe_float(r.get(sret_col)) if sret_col in rank.columns else float("nan")
        ssh = _safe_float(r.get(ssh_col)) if ssh_col in rank.columns else float("nan")
        cal_ece = _safe_float(r.get("cal_ece"))
        flags = str(r.get("principle_flags") or "none")

        if decision == "promote":
            action = "Approve for controlled deployment (paper/live-light)."
        elif decision == "candidate":
            action = "Keep in candidate queue; tighten ML validation and monitor drift."
        else:
            action = "Do not deploy; refit/research required."

        lines.append(f"### {i + 1}. `{run}` — **{decision.upper()}**")
        lines.append("")
        lines.append("- **Scorecard**")
        lines.append(f"  - Promotion score: `{score:.4f}`")
        lines.append(f"  - Reliability score: `{rel:.4f}`")
        lines.append(f"  - ML quality score: `{mlq:.4f}`")
        lines.append("- **Finance profile**")
        lines.append(f"  - Sharpe: `{sharpe:.4f}`")
        lines.append(f"  - Max drawdown: `{dd:.4f}`")
        lines.append(f"  - Trades: `{int(ntr) if np.isfinite(ntr) else 0}`")
        lines.append(f"  - Profit factor: `{pf:.4f}`")
        lines.append("- **Stress & calibration checks**")
        lines.append(f"  - Stress total return @ `{sb}` bps: `{sret:.4f}`")
        lines.append(f"  - Stress Sharpe @ `{sb}` bps: `{ssh:.4f}`")
        lines.append(f"  - Calibration ECE: `{cal_ece:.4f}`")
        lines.append("- **Risk flags**")
        lines.append(f"  - Principle flags: `{flags}`")
        lines.append("- **Committee recommendation**")
        lines.append(f"  - {action}")
        lines.append("")

    # ---- Deployment checklist (auto-suggested thresholds) ----
    base = promoted if not promoted.empty else rank
    base_dd = pd.to_numeric(base.get("m_max_drawdown"), errors="coerce") if "m_max_drawdown" in base.columns else pd.Series(dtype=float)
    base_sh = pd.to_numeric(base.get("m_sharpe"), errors="coerce") if "m_sharpe" in base.columns else pd.Series(dtype=float)
    base_pf = pd.to_numeric(base.get("t_profit_factor"), errors="coerce") if "t_profit_factor" in base.columns else pd.Series(dtype=float)
    base_rel = pd.to_numeric(base.get("reliability_score"), errors="coerce") if "reliability_score" in base.columns else pd.Series(dtype=float)
    base_mlq = pd.to_numeric(base.get("ml_quality_score"), errors="coerce") if "ml_quality_score" in base.columns else pd.Series(dtype=float)
    base_cal = pd.to_numeric(base.get("cal_ece"), errors="coerce") if "cal_ece" in base.columns else pd.Series(dtype=float)

    dd_abs_med = float(np.nanmedian(np.abs(base_dd))) if not base_dd.empty else float("nan")
    sh_med = float(np.nanmedian(base_sh)) if not base_sh.empty else float("nan")
    pf_med = float(np.nanmedian(base_pf)) if not base_pf.empty else float("nan")
    rel_med = float(np.nanmedian(base_rel)) if not base_rel.empty else float("nan")
    mlq_med = float(np.nanmedian(base_mlq)) if not base_mlq.empty else float("nan")
    cal_med = float(np.nanmedian(base_cal)) if not base_cal.empty else float("nan")

    # Conservative default bounds if sample is sparse.
    hard_dd = 0.25 if not np.isfinite(dd_abs_med) else float(np.clip(dd_abs_med * 1.25, 0.18, 0.35))
    soft_dd = float(np.clip(hard_dd * 0.70, 0.10, 0.25))
    min_sharpe_live = 0.60 if not np.isfinite(sh_med) else float(np.clip(sh_med * 0.75, 0.50, 1.20))
    min_pf_live = 1.10 if not np.isfinite(pf_med) else float(np.clip(pf_med * 0.85, 1.05, 1.50))
    min_rel_live = 0.60 if not np.isfinite(rel_med) else float(np.clip(rel_med * 0.90, 0.55, 0.90))
    min_mlq_live = 0.55 if not np.isfinite(mlq_med) else float(np.clip(mlq_med * 0.90, 0.50, 0.90))
    max_ece_live = 0.12 if not np.isfinite(cal_med) else float(np.clip(cal_med * 1.30, 0.08, 0.20))

    lines.append("## Deployment checklist")
    lines.append("")
    lines.append("### 1) Portfolio construction limits")
    lines.append("")
    lines.append("- [ ] Cap per-strategy gross exposure at **10–20%** of total risk budget during rollout.")
    lines.append("- [ ] Start with **paper trading** or de minimis capital for at least one full retrain cycle.")
    lines.append(f"- [ ] Enforce hard max drawdown per strategy: **{hard_dd:.2%}** (soft warning at **{soft_dd:.2%}**).")
    lines.append("")
    lines.append("### 2) Kill-switch thresholds")
    lines.append("")
    lines.append(f"- [ ] Pause strategy if trailing live Sharpe < **{min_sharpe_live:.2f}** over review window.")
    lines.append(f"- [ ] Pause strategy if profit factor < **{min_pf_live:.2f}**.")
    lines.append(f"- [ ] Pause strategy if stress Sharpe @ **{sb} bps** drops below **{min_sharpe_live:.2f}**.")
    lines.append(f"- [ ] Pause strategy if reliability score < **{min_rel_live:.2f}** or ML quality < **{min_mlq_live:.2f}**.")
    lines.append("")
    lines.append("### 3) Retrain and promotion cadence")
    lines.append("")
    lines.append("- [ ] Retrain on a fixed cadence (weekly for fast markets, monthly for slower daily bars).")
    lines.append("- [ ] Re-run walk-forward validation before every promotion event.")
    lines.append("- [ ] Require unchanged pass status in promotion gate across **2 consecutive runs** before scaling allocation.")
    lines.append("")
    lines.append("### 4) Monitoring KPIs")
    lines.append("")
    lines.append("- [ ] Daily: realized PnL, turnover, slippage drift vs assumed bps.")
    lines.append("- [ ] Weekly: Sharpe, max drawdown, profit factor, win rate.")
    lines.append(f"- [ ] Model health: calibration ECE ≤ **{max_ece_live:.3f}** (lower is better), drift in class balance/probabilities.")
    lines.append("- [ ] Governance: track decision state (`promote/candidate/reject`) and record reasons for overrides.")
    lines.append("")
    lines.append("### 5) Operational controls")
    lines.append("")
    lines.append("- [ ] Pre-trade checks: stale data guard, max spread/slippage guard, venue health check.")
    lines.append("- [ ] Intraday emergency stop: disable new entries on data outages or order rejection spikes.")
    lines.append("- [ ] Post-trade audit log: persist signal inputs, model version, and execution metadata.")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This memo is ranking-oriented, not a guarantee of future performance.")
    lines.append("- Use alongside the dashboard and raw CSV reports for final review.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_cleanup_plan(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for d in _discover_dirs(root):
        cat = _classify_dir(d.name)
        if cat == "other":
            continue

        target_rel = _target_for_category(cat) / d.name
        # already organized if this directory is itself under artifacts/* and named same.
        if str(d).replace("\\", "/").startswith(str((root / "artifacts")).replace("\\", "/")):
            continue

        rows.append(
            {
                "source": str(d),
                "category": cat,
                "target": str(root / target_rel),
                "action": "move",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["source", "category", "target", "action"])

    plan = pd.DataFrame(rows)
    return plan.sort_values(["category", "source"]).reset_index(drop=True)


def _apply_cleanup(plan: pd.DataFrame, *, max_moves: int) -> tuple[int, list[str]]:
    moved = 0
    notes: list[str] = []

    if plan is None or plan.empty:
        return moved, notes

    for _, row in plan.iterrows():
        if moved >= max_moves:
            notes.append(f"stopped_after_max_moves={max_moves}")
            break

        src = Path(str(row["source"]))
        dst = Path(str(row["target"]))

        if not src.exists():
            notes.append(f"skip_missing:{src}")
            continue
        if dst.exists():
            notes.append(f"skip_target_exists:{dst}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            notes.append(f"move_failed:{src}->{dst}:{e}")

    return moved, notes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Master-brain governance: finance/ML checks + folder cleanup planning")
    p.add_argument("--in-root", default=".", help="Project root to scan")
    p.add_argument("--out-dir", default="master_brain", help="Where to write master-brain outputs")

    p.add_argument("--min-trades", type=int, default=30, help="Finance policy: minimum trades for confidence")
    p.add_argument("--min-sharpe", type=float, default=0.5, help="Finance policy: minimum Sharpe threshold")
    p.add_argument(
        "--max-drawdown-abs",
        type=float,
        default=0.35,
        help="Finance policy: max allowed absolute drawdown (0.35 = 35%)",
    )
    p.add_argument("--min-profit-factor", type=float, default=1.05, help="Finance policy: minimum profit factor")
    p.add_argument("--min-cv-folds", type=int, default=4, help="ML policy: minimum walk-forward folds")
    p.add_argument("--min-reliability-score", type=float, default=0.60, help="Promotion gate: minimum reliability score")
    p.add_argument("--min-ml-quality-score", type=float, default=0.55, help="Promotion gate: minimum ML quality score")

    p.add_argument(
        "--stress-bps-grid",
        default="10,20,35",
        help="Extra cost stress grid in bps, comma-separated (e.g., 10,20,35)",
    )
    p.add_argument("--gate-stress-bps", type=float, default=20.0, help="Stress level (bps) used for promotion gate")
    p.add_argument("--memo-top-n", type=int, default=10, help="How many runs to include in top promotion memo")

    p.add_argument("--apply-cleanup", action="store_true", help="Actually move directories per cleanup plan")
    p.add_argument("--max-moves", type=int, default=150, help="Safety cap when --apply-cleanup is used")

    args = p.parse_args(argv)

    root = Path(args.in_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = Policy(
        min_trades=int(args.min_trades),
        min_sharpe=float(args.min_sharpe),
        max_drawdown_abs=float(args.max_drawdown_abs),
        min_profit_factor=float(args.min_profit_factor),
        min_cv_folds=int(args.min_cv_folds),
        min_reliability_score=float(args.min_reliability_score),
        min_ml_quality_score=float(args.min_ml_quality_score),
    )

    bt_df = _scan_backtests(root, policy=policy)
    ml_df = _scan_ml(root, policy=policy)
    stress_grid = _parse_bps_grid(args.stress_bps_grid)
    bt_df = _apply_cost_stress(bt_df, stress_bps_grid=stress_grid)
    gate_df = _build_promotion_gate(
        bt_df,
        ml_df,
        policy=policy,
        gate_stress_bps=float(args.gate_stress_bps),
    )
    cleanup_plan = _build_cleanup_plan(root)

    bt_csv = out_dir / "finance_principles_report.csv"
    ml_csv = out_dir / "ml_principles_report.csv"
    cp_csv = out_dir / "folder_cleanup_plan.csv"
    gate_csv = out_dir / "promotion_gate_report.csv"
    dash_html = out_dir / "master_brain_dashboard.html"
    memo_md = out_dir / "top_promotion_memo.md"

    if bt_df.empty:
        pd.DataFrame(columns=[
            "run",
            "reliability_score",
            "m_sharpe",
            "m_max_drawdown",
            "m_total_return",
            "t_n_trades",
            "t_profit_factor",
            "principle_flags",
        ]).to_csv(bt_csv, index=False)
    else:
        bt_df.to_csv(bt_csv, index=False)

    if ml_df.empty:
        pd.DataFrame(columns=["run", "ml_quality_score", "cv_present", "n_folds", "cv_metric", "cv_mean", "cv_std"]).to_csv(ml_csv, index=False)
    else:
        ml_df.to_csv(ml_csv, index=False)

    if gate_df.empty:
        pd.DataFrame(columns=["run", "decision", "promotion_score", "pass_finance", "pass_stress", "pass_ml"]).to_csv(gate_csv, index=False)
    else:
        gate_df.to_csv(gate_csv, index=False)

    _write_phase2_dashboard(
        bt_df,
        ml_df,
        gate_df,
        stress_grid=stress_grid,
        gate_stress_bps=float(args.gate_stress_bps),
        out_html=dash_html,
    )

    _write_promotion_memo(
        gate_df,
        out_md=memo_md,
        gate_stress_bps=float(args.gate_stress_bps),
        top_n=int(args.memo_top_n),
    )

    cleanup_plan.to_csv(cp_csv, index=False)

    moved = 0
    apply_notes: list[str] = []
    if args.apply_cleanup:
        moved, apply_notes = _apply_cleanup(cleanup_plan, max_moves=int(args.max_moves))

    best_bt = bt_df.head(1).to_dict(orient="records") if not bt_df.empty else []
    best_ml = ml_df.head(1).to_dict(orient="records") if not ml_df.empty else []
    promoted = gate_df[gate_df["decision"] == "promote"] if (not gate_df.empty and "decision" in gate_df.columns) else pd.DataFrame()
    top_promoted = promoted.head(3).to_dict(orient="records") if not promoted.empty else []

    summary = {
        "root": str(root),
        "policy": {
            "min_trades": policy.min_trades,
            "min_sharpe": policy.min_sharpe,
            "max_drawdown_abs": policy.max_drawdown_abs,
            "min_profit_factor": policy.min_profit_factor,
            "min_cv_folds": policy.min_cv_folds,
            "min_reliability_score": policy.min_reliability_score,
            "min_ml_quality_score": policy.min_ml_quality_score,
            "stress_bps_grid": stress_grid,
            "gate_stress_bps": float(args.gate_stress_bps),
        },
        "counts": {
            "backtests_scored": int(0 if bt_df.empty else len(bt_df)),
            "ml_runs_scored": int(0 if ml_df.empty else len(ml_df)),
            "promoted_runs": int(0 if promoted is None or promoted.empty else len(promoted)),
            "cleanup_moves_planned": int(len(cleanup_plan)),
            "cleanup_moves_applied": int(moved),
        },
        "best_backtest": best_bt,
        "best_ml": best_ml,
        "top_promoted": top_promoted,
        "outputs": {
            "finance_principles_report_csv": str(bt_csv),
            "ml_principles_report_csv": str(ml_csv),
            "promotion_gate_report_csv": str(gate_csv),
            "master_brain_dashboard_html": str(dash_html),
            "top_promotion_memo_md": str(memo_md),
            "folder_cleanup_plan_csv": str(cp_csv),
        },
        "apply_cleanup": bool(args.apply_cleanup),
        "apply_notes": apply_notes,
    }

    summary_json = out_dir / "master_brain_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Human-readable executive summary.
    md = out_dir / "master_brain_summary.md"
    lines: list[str] = []
    lines.append("# Master Brain Summary")
    lines.append("")
    lines.append("## Top findings")
    lines.append("")
    lines.append(f"- Backtests scored: **{summary['counts']['backtests_scored']}**")
    lines.append(f"- ML runs scored: **{summary['counts']['ml_runs_scored']}**")
    lines.append(f"- Runs promoted by gate: **{summary['counts']['promoted_runs']}**")
    lines.append(f"- Cleanup moves planned: **{summary['counts']['cleanup_moves_planned']}**")
    lines.append(f"- Cleanup moves applied: **{summary['counts']['cleanup_moves_applied']}**")
    lines.append("")

    if best_bt:
        b = best_bt[0]
        lines.append("## Best backtest (risk-adjusted)")
        lines.append("")
        lines.append(f"- Run: `{b.get('run')}`")
        lines.append(f"- Reliability score: `{_safe_float(b.get('reliability_score')):.4f}`")
        lines.append(f"- Sharpe: `{_safe_float(b.get('m_sharpe')):.4f}`")
        lines.append(f"- Max drawdown: `{_safe_float(b.get('m_max_drawdown')):.4f}`")
        lines.append(f"- Profit factor: `{_safe_float(b.get('t_profit_factor')):.4f}`")
        lines.append(f"- Principle flags: `{b.get('principle_flags') or 'none'}`")
        lines.append("")

    if best_ml:
        b = best_ml[0]
        lines.append("## Best ML run (stability-aware)")
        lines.append("")
        lines.append(f"- Run: `{b.get('run')}`")
        lines.append(f"- ML quality score: `{_safe_float(b.get('ml_quality_score')):.4f}`")
        lines.append(f"- CV metric: `{b.get('cv_metric')}`")
        lines.append(f"- CV mean: `{_safe_float(b.get('cv_mean')):.4f}`")
        lines.append(f"- CV std: `{_safe_float(b.get('cv_std')):.4f}`")
        lines.append("")

    if top_promoted:
        lines.append("## Promotion gate picks")
        lines.append("")
        for i, r in enumerate(top_promoted, start=1):
            lines.append(
                f"{i}. `{r.get('run')}` | score=`{_safe_float(r.get('promotion_score')):.4f}` | reliability=`{_safe_float(r.get('reliability_score')):.4f}` | ml_quality=`{_safe_float(r.get('ml_quality_score')):.4f}`"
            )
        lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append(f"- `{bt_csv}`")
    lines.append(f"- `{ml_csv}`")
    lines.append(f"- `{gate_csv}`")
    lines.append(f"- `{dash_html}`")
    lines.append(f"- `{memo_md}`")
    lines.append(f"- `{cp_csv}`")
    lines.append(f"- `{summary_json}`")

    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {bt_csv}")
    print(f"Wrote: {ml_csv}")
    print(f"Wrote: {gate_csv}")
    print(f"Wrote: {dash_html}")
    print(f"Wrote: {memo_md}")
    print(f"Wrote: {cp_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {md}")
    if args.apply_cleanup:
        print(f"Applied cleanup moves: {moved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
