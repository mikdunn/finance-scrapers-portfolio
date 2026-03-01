"""Master-brain governance pipeline.

Speed-focused design:
- Parallel directory scans for backtest/ML runs.
- Vectorized stress calculations in pandas/numpy.
- Single-pass report generation.

Outputs:
- finance_principles_report.csv
- ml_principles_report.csv
- promotion_gate_report.csv
- monitoring_metrics_report.csv
- quality_metrics_report.csv
- master_brain_summary.json
- master_brain_summary.md

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
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


def _numeric_series(df: pd.DataFrame, col: str, *, default: float = float("nan")) -> pd.Series:
    s = df[col] if col in df.columns else pd.Series(default, index=df.index)
    return pd.to_numeric(s, errors="coerce")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    return sorted(set(vals))


def _annualized_score(sharpe: float, drawdown: float, cagr: float) -> float:
    s = float(np.tanh(np.nan_to_num(sharpe, nan=0.0) / 2.0))
    dd_abs = abs(drawdown) if np.isfinite(drawdown) else 1.0
    d = float(np.clip(1.0 - dd_abs / 0.6, 0.0, 1.0))
    g = float(np.tanh(np.nan_to_num(cagr, nan=0.0) / 0.35))
    return 0.50 * s + 0.35 * d + 0.15 * g


def _trade_quality_score(win_rate: float, profit_factor: float, n_trades: int, min_trades: int) -> float:
    wr = np.nan_to_num(win_rate, nan=0.0)
    pf = np.nan_to_num(profit_factor, nan=0.0)
    wr_term = float(np.clip((wr - 0.45) / 0.20, 0.0, 1.0))
    pf_term = float(np.clip((pf - 1.0) / 1.0, 0.0, 1.0))
    sample_term = float(np.clip(n_trades / max(1.0, float(min_trades)), 0.0, 1.0))
    return 0.35 * wr_term + 0.45 * pf_term + 0.20 * sample_term


def _model_run_from_model_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    try:
        p = Path(str(model_path))
        return p.parent.name if p.parent is not None else None
    except Exception:
        return None


def _load_backtest_row(bt_dir: Path, policy: Policy) -> dict | None:
    summary_path = bt_dir / "backtest_summary.json"
    if not summary_path.exists():
        return None

    try:
        s = _read_json(summary_path)
    except Exception:
        return None

    metrics = s.get("portfolio_metrics") if isinstance(s.get("portfolio_metrics"), dict) else s.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    trade_metrics = s.get("trade_metrics") if isinstance(s.get("trade_metrics"), dict) else {}
    if trade_metrics is None:
        trade_metrics = {}

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
    n_trades_raw = _safe_float(trade_metrics.get("n_trades"))
    n_trades = int(n_trades_raw) if np.isfinite(n_trades_raw) else 0

    finance_score = _annualized_score(sharpe=sharpe, drawdown=dd, cagr=cagr)
    trade_score = _trade_quality_score(win_rate=win_rate, profit_factor=profit_factor, n_trades=n_trades, min_trades=policy.min_trades)

    flags: list[str] = []
    if np.isfinite(sharpe) and sharpe < policy.min_sharpe:
        flags.append("low_sharpe")
    if np.isfinite(dd) and abs(dd) > policy.max_drawdown_abs:
        flags.append("drawdown_too_high")
    if n_trades > 0 and n_trades < policy.min_trades:
        flags.append("too_few_trades")
    if np.isfinite(profit_factor) and profit_factor < policy.min_profit_factor:
        flags.append("weak_profit_factor")

    penalty = float(np.clip(0.08 * len(flags), 0.0, 0.35))
    reliability = float(np.clip(0.65 * finance_score + 0.35 * trade_score - penalty, 0.0, 1.0))

    return {
        "run": bt_dir.name,
        "path": str(bt_dir),
        "asset": s.get("asset"),
        "model": s.get("model"),
        "model_run": _model_run_from_model_path(s.get("model")),
        "mode": s.get("mode"),
        "signal_source": s.get("signal_source"),
        "portfolio_weighting": s.get("portfolio_weighting"),
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
        "principle_flags": ",".join(flags),
        "n_principle_flags": len(flags),
    }


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
    fold_term = float(np.clip(len(cv) / max(1.0, float(policy.min_cv_folds)), 0.0, 1.0))

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
        quality = float(1.0 / (1.0 + max(mean, 0.0)))
        stability = float(np.clip(1.0 - std / max(1e-9, mean + 1e-9), 0.0, 1.0))
        score = 0.35 * fold_term + 0.45 * quality + 0.20 * stability
        return score, {**out, "cv_metric": "rmse", "cv_mean": mean, "cv_std": std}

    return 0.0, {**out, "cv_metric": "unknown"}


def _calibration_quality_from_metrics(m: dict) -> tuple[float, dict]:
    if not isinstance(m, dict):
        return 0.0, {"cal_present": False}

    ece = _safe_float(m.get("cal_ece"))
    brier = _safe_float(m.get("cal_brier_multi"))
    logloss = _safe_float(m.get("cal_log_loss"))

    if not any(np.isfinite(x) for x in (ece, brier, logloss)):
        return 0.0, {"cal_present": False}

    ece_term = float(np.clip(1.0 - (ece / 0.25), 0.0, 1.0)) if np.isfinite(ece) else float("nan")
    brier_term = float(np.clip(1.0 - (brier / 0.75), 0.0, 1.0)) if np.isfinite(brier) else float("nan")
    ll_term = float(1.0 / (1.0 + max(logloss, 0.0))) if np.isfinite(logloss) else float("nan")
    terms = [x for x in (ece_term, brier_term, ll_term) if np.isfinite(x)]
    if not terms:
        return 0.0, {"cal_present": False}

    return float(np.mean(terms)), {
        "cal_present": True,
        "cal_ece": ece,
        "cal_brier_multi": brier,
        "cal_log_loss": logloss,
    }


def _scan_backtests(root: Path, policy: Policy, *, workers: int) -> pd.DataFrame:
    backtest_dirs = [p for p in root.iterdir() if p.is_dir() and _classify_dir(p.name) == "backtests"]
    artifacts_backtests = root / "artifacts" / "backtests"
    if artifacts_backtests.exists() and artifacts_backtests.is_dir():
        backtest_dirs.extend([p for p in artifacts_backtests.iterdir() if p.is_dir()])

    seen: set[str] = set()
    uniq_dirs: list[Path] = []
    for d in backtest_dirs:
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq_dirs.append(d)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for row in ex.map(lambda d: _load_backtest_row(d, policy), uniq_dirs):
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["reliability_score", "m_sharpe", "m_total_return"], ascending=False).reset_index(drop=True)


def _scan_ml(root: Path, policy: Policy, *, workers: int) -> pd.DataFrame:
    ml_dirs = [p for p in root.iterdir() if p.is_dir() and _classify_dir(p.name) == "ml"]
    artifacts_ml = root / "artifacts" / "ml"
    if artifacts_ml.exists() and artifacts_ml.is_dir():
        ml_dirs.extend([p for p in artifacts_ml.iterdir() if p.is_dir()])

    seen: set[str] = set()
    uniq_dirs: list[Path] = []
    for d in ml_dirs:
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq_dirs.append(d)

    def _one(d: Path) -> dict:
        m = {}
        mp = d / "metrics.json"
        if mp.exists():
            try:
                m = _read_json(mp)
            except Exception:
                m = {}
        cv_score, cv_meta = _cv_quality_score(d, policy)
        cal_score, cal_meta = _calibration_quality_from_metrics(m)
        ml_quality = float(np.clip(0.55 * cv_score + 0.45 * cal_score, 0.0, 1.0)) if bool(cal_meta.get("cal_present")) else float(np.clip(cv_score, 0.0, 1.0))
        return {
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

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for row in ex.map(_one, uniq_dirs):
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ml_quality_score", "run"], ascending=[False, True]).reset_index(drop=True)


def _apply_cost_stress(bt_df: pd.DataFrame, *, stress_bps_grid: list[float]) -> pd.DataFrame:
    if bt_df is None or bt_df.empty:
        return bt_df

    out = bt_df.copy()
    trn = _numeric_series(out, "m_turnover_sum")
    n_bars = _numeric_series(out, "m_n_bars")
    avg = _numeric_series(out, "m_avg_bar_return")
    vol = _numeric_series(out, "m_vol_bar")
    ann = _numeric_series(out, "m_ann_factor")
    total = _numeric_series(out, "m_total_return")

    for bps in stress_bps_grid:
        b = int(round(float(bps)))
        rate = float(bps) / 10000.0
        extra_drag = rate * trn
        out[f"stress_total_return_bps{b}"] = total - extra_drag

        per_bar_drag = np.where((n_bars > 0) & np.isfinite(n_bars), extra_drag / n_bars, np.nan)
        avg_new = avg - per_bar_drag
        sharpe_new = np.where((vol > 0) & np.isfinite(vol) & (ann > 0) & np.isfinite(ann), (avg_new / vol) * np.sqrt(ann), np.nan)
        out[f"stress_sharpe_bps{b}"] = sharpe_new

    return out


def _build_promotion_gate(bt_df: pd.DataFrame, ml_df: pd.DataFrame, *, policy: Policy, gate_stress_bps: float) -> pd.DataFrame:
    if bt_df is None or bt_df.empty:
        return pd.DataFrame()

    out = bt_df.copy()
    if ml_df is not None and not ml_df.empty and "run" in ml_df.columns:
        keep_cols = [c for c in ["run", "ml_quality_score", "cv_present", "n_folds", "cv_metric", "cv_mean", "cv_std", "calibration_quality_score", "cal_ece", "cal_brier_multi", "cal_log_loss"] if c in ml_df.columns]
        if keep_cols:
            out = out.merge(ml_df[keep_cols].rename(columns={"run": "model_run"}), on="model_run", how="left")

    sb = int(round(float(gate_stress_bps)))
    sret_col = f"stress_total_return_bps{sb}"
    ssh_col = f"stress_sharpe_bps{sb}"

    sret = _numeric_series(out, sret_col)
    ssh = _numeric_series(out, ssh_col)
    mlq = _numeric_series(out, "ml_quality_score")
    rel = _numeric_series(out, "reliability_score")
    flags = _numeric_series(out, "n_principle_flags", default=0.0).fillna(0)
    cv_present = out.get("cv_present", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    out["pass_finance"] = (rel >= float(policy.min_reliability_score)) & (flags <= 0)
    out["pass_stress"] = (sret > 0.0) & (ssh >= float(policy.min_sharpe))
    out["pass_ml"] = (mlq >= float(policy.min_ml_quality_score)) & cv_present

    out["promotion_score"] = 0.55 * np.nan_to_num(rel, nan=0.0) + 0.30 * np.nan_to_num(mlq, nan=0.0) + 0.15 * np.clip(np.nan_to_num(ssh / 2.0, nan=0.0), 0.0, 1.0)
    out["decision"] = np.where(out["pass_finance"] & out["pass_stress"] & out["pass_ml"], "promote", np.where(out["pass_finance"] & out["pass_stress"], "candidate", "reject"))

    return out.sort_values(["decision", "promotion_score"], ascending=[True, False]).reset_index(drop=True)


def _build_monitoring_metrics(gate_df: pd.DataFrame, *, gate_stress_bps: float) -> pd.DataFrame:
    if gate_df is None or gate_df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    sb = int(round(float(gate_stress_bps)))
    sret_col = f"stress_total_return_bps{sb}"
    ssh_col = f"stress_sharpe_bps{sb}"

    df = gate_df.copy()
    for c in ["promotion_score", "reliability_score", "ml_quality_score", "m_sharpe", "m_max_drawdown", "t_profit_factor", sret_col, ssh_col, "cal_ece"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    metrics = {
        "runs_total": float(len(df)),
        "runs_promote": float((df["decision"] == "promote").sum()) if "decision" in df.columns else 0.0,
        "runs_candidate": float((df["decision"] == "candidate").sum()) if "decision" in df.columns else 0.0,
        "runs_reject": float((df["decision"] == "reject").sum()) if "decision" in df.columns else 0.0,
        "promotion_score_median": float(df["promotion_score"].median()) if "promotion_score" in df.columns else float("nan"),
        "reliability_score_median": float(df["reliability_score"].median()) if "reliability_score" in df.columns else float("nan"),
        "ml_quality_score_median": float(df["ml_quality_score"].median()) if "ml_quality_score" in df.columns else float("nan"),
        "sharpe_median": float(df["m_sharpe"].median()) if "m_sharpe" in df.columns else float("nan"),
        "max_drawdown_median": float(df["m_max_drawdown"].median()) if "m_max_drawdown" in df.columns else float("nan"),
        "profit_factor_median": float(df["t_profit_factor"].median()) if "t_profit_factor" in df.columns else float("nan"),
        f"stress_total_return_median_bps{sb}": float(df[sret_col].median()) if sret_col in df.columns else float("nan"),
        f"stress_sharpe_median_bps{sb}": float(df[ssh_col].median()) if ssh_col in df.columns else float("nan"),
        "cal_ece_median": float(df["cal_ece"].median()) if "cal_ece" in df.columns else float("nan"),
    }

    return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})


def _build_cleanup_plan(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    artifacts_root = str((root / "artifacts").resolve()).replace("\\", "/")
    for d in root.iterdir():
        if not d.is_dir():
            continue
        cat = _classify_dir(d.name)
        if cat == "other":
            continue
        if str(d.resolve()).replace("\\", "/").startswith(artifacts_root):
            continue
        rows.append({"source": str(d), "category": cat, "target": str(root / _target_for_category(cat) / d.name), "action": "move"})
    if not rows:
        return pd.DataFrame(columns=["source", "category", "target", "action"])
    return pd.DataFrame(rows).sort_values(["category", "source"]).reset_index(drop=True)


def _execute_cleanup_plan(cleanup_df: pd.DataFrame, *, mode: str, dry_run: bool) -> pd.DataFrame:
    if cleanup_df is None or cleanup_df.empty:
        return pd.DataFrame(columns=["source", "target", "category", "mode", "status", "detail"])

    rows: list[dict] = []
    for r in cleanup_df.to_dict(orient="records"):
        source = str(r.get("source", ""))
        target = str(r.get("target", ""))
        category = str(r.get("category", ""))
        status = "planned"
        detail = ""

        try:
            src = Path(source)
            if not src.exists():
                status = "skipped"
                detail = "source_missing"
            elif dry_run:
                status = "dry_run"
                detail = "no_changes"
            elif mode == "delete":
                shutil.rmtree(src)
                status = "deleted"
                detail = "ok"
            else:
                dst = Path(target)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    base_name = dst.name
                    i = 1
                    while True:
                        cand = dst.parent / f"{base_name}__moved_{i}"
                        if not cand.exists():
                            dst = cand
                            break
                        i += 1
                shutil.move(str(src), str(dst))
                target = str(dst)
                status = "moved"
                detail = "ok"
        except Exception as e:
            status = "error"
            detail = str(e)

        rows.append({
            "source": source,
            "target": target,
            "category": category,
            "mode": mode,
            "status": status,
            "detail": detail,
        })

    return pd.DataFrame(rows)


def _build_output_quality_metrics(bt_df: pd.DataFrame, ml_df: pd.DataFrame, gate_df: pd.DataFrame, *, gate_stress_bps: float) -> pd.DataFrame:
    sb = int(round(float(gate_stress_bps)))
    sret_col = f"stress_total_return_bps{sb}"
    ssh_col = f"stress_sharpe_bps{sb}"

    metrics: dict[str, float] = {}

    bt_n = float(len(bt_df)) if bt_df is not None and not bt_df.empty else 0.0
    ml_n = float(len(ml_df)) if ml_df is not None and not ml_df.empty else 0.0
    gate_n = float(len(gate_df)) if gate_df is not None and not gate_df.empty else 0.0

    metrics["backtests_scored"] = bt_n
    metrics["ml_runs_scored"] = ml_n
    metrics["gated_runs"] = gate_n

    if bt_n > 0:
        rel = _numeric_series(bt_df, "reliability_score")
        flags = _numeric_series(bt_df, "n_principle_flags")
        metrics["reliability_non_null_rate"] = float(rel.notna().mean())
        metrics["reliability_mean"] = float(rel.mean()) if rel.notna().any() else float("nan")
        metrics["principles_clean_rate"] = float((flags.fillna(9999) <= 0).mean())

    if ml_n > 0:
        mlq = _numeric_series(ml_df, "ml_quality_score")
        cvp = ml_df.get("cv_present", pd.Series(False, index=ml_df.index)).fillna(False).astype(bool)
        metrics["ml_quality_non_null_rate"] = float(mlq.notna().mean())
        metrics["ml_quality_mean"] = float(mlq.mean()) if mlq.notna().any() else float("nan")
        metrics["cv_present_rate"] = float(cvp.mean())

    if gate_n > 0:
        promo = _numeric_series(gate_df, "promotion_score")
        p_fin = gate_df.get("pass_finance", pd.Series(False, index=gate_df.index)).fillna(False).astype(bool)
        p_stress = gate_df.get("pass_stress", pd.Series(False, index=gate_df.index)).fillna(False).astype(bool)
        p_ml = gate_df.get("pass_ml", pd.Series(False, index=gate_df.index)).fillna(False).astype(bool)

        metrics["promotion_score_non_null_rate"] = float(promo.notna().mean())
        metrics["promotion_score_mean"] = float(promo.mean()) if promo.notna().any() else float("nan")
        metrics["pass_finance_rate"] = float(p_fin.mean())
        metrics["pass_stress_rate"] = float(p_stress.mean())
        metrics["pass_ml_rate"] = float(p_ml.mean())

        if sret_col in gate_df.columns:
            sret = pd.to_numeric(gate_df[sret_col], errors="coerce")
            metrics[f"{sret_col}_mean"] = float(sret.mean()) if sret.notna().any() else float("nan")
        if ssh_col in gate_df.columns:
            ssh = pd.to_numeric(gate_df[ssh_col], errors="coerce")
            metrics[f"{ssh_col}_mean"] = float(ssh.mean()) if ssh.notna().any() else float("nan")

    return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Master-brain governance: speed + quality + monitoring")
    p.add_argument("--in-root", default=".")
    p.add_argument("--out-dir", default="master_brain")

    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--min-sharpe", type=float, default=0.5)
    p.add_argument("--max-drawdown-abs", type=float, default=0.35)
    p.add_argument("--min-profit-factor", type=float, default=1.05)
    p.add_argument("--min-cv-folds", type=int, default=4)
    p.add_argument("--min-reliability-score", type=float, default=0.60)
    p.add_argument("--min-ml-quality-score", type=float, default=0.55)

    p.add_argument("--stress-bps-grid", default="10,20,35")
    p.add_argument("--gate-stress-bps", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=max(2, os.cpu_count() or 2))
    p.add_argument("--apply-cleanup", action="store_true", help="Apply folder cleanup actions before scoring")
    p.add_argument("--cleanup-mode", choices=["move", "delete"], default="move", help="Cleanup behavior: move to artifacts/* or delete permanently")
    p.add_argument("--cleanup-dry-run", action="store_true", help="Plan cleanup actions without changing folders")

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

    workers = max(1, int(args.workers))
    stress_grid = _parse_bps_grid(args.stress_bps_grid)

    cleanup_df_before = _build_cleanup_plan(root)
    cleanup_exec_df = _execute_cleanup_plan(cleanup_df_before, mode=args.cleanup_mode, dry_run=bool(args.cleanup_dry_run)) if args.apply_cleanup else pd.DataFrame(columns=["source", "target", "category", "mode", "status", "detail"])
    cleanup_df_after = _build_cleanup_plan(root)

    bt_df = _scan_backtests(root, policy=policy, workers=workers)
    ml_df = _scan_ml(root, policy=policy, workers=workers)
    bt_df = _apply_cost_stress(bt_df, stress_bps_grid=stress_grid)
    gate_df = _build_promotion_gate(bt_df, ml_df, policy=policy, gate_stress_bps=float(args.gate_stress_bps))
    mon_df = _build_monitoring_metrics(gate_df, gate_stress_bps=float(args.gate_stress_bps))
    quality_df = _build_output_quality_metrics(bt_df, ml_df, gate_df, gate_stress_bps=float(args.gate_stress_bps))

    bt_csv = out_dir / "finance_principles_report.csv"
    ml_csv = out_dir / "ml_principles_report.csv"
    gate_csv = out_dir / "promotion_gate_report.csv"
    mon_csv = out_dir / "monitoring_metrics_report.csv"
    cp_csv = out_dir / "folder_cleanup_plan.csv"
    cx_csv = out_dir / "folder_cleanup_execution_report.csv"
    cq_csv = out_dir / "folder_cleanup_remaining_plan.csv"
    q_csv = out_dir / "quality_metrics_report.csv"

    (bt_df if not bt_df.empty else pd.DataFrame(columns=["run", "reliability_score"])).to_csv(bt_csv, index=False)
    (ml_df if not ml_df.empty else pd.DataFrame(columns=["run", "ml_quality_score"])).to_csv(ml_csv, index=False)
    (gate_df if not gate_df.empty else pd.DataFrame(columns=["run", "decision", "promotion_score"])).to_csv(gate_csv, index=False)
    mon_df.to_csv(mon_csv, index=False)
    cleanup_df_before.to_csv(cp_csv, index=False)
    cleanup_exec_df.to_csv(cx_csv, index=False)
    cleanup_df_after.to_csv(cq_csv, index=False)
    quality_df.to_csv(q_csv, index=False)

    top_promoted = gate_df[gate_df["decision"] == "promote"].head(5).to_dict(orient="records") if (not gate_df.empty and "decision" in gate_df.columns) else []

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
            "workers": workers,
            "apply_cleanup": bool(args.apply_cleanup),
            "cleanup_mode": str(args.cleanup_mode),
            "cleanup_dry_run": bool(args.cleanup_dry_run),
        },
        "counts": {
            "backtests_scored": int(len(bt_df)) if not bt_df.empty else 0,
            "ml_runs_scored": int(len(ml_df)) if not ml_df.empty else 0,
            "promoted_runs": int((gate_df["decision"] == "promote").sum()) if (not gate_df.empty and "decision" in gate_df.columns) else 0,
            "cleanup_moves_planned": int(len(cleanup_df_before)),
            "cleanup_actions_executed": int((cleanup_exec_df["status"].isin(["moved", "deleted"]).sum())) if not cleanup_exec_df.empty else 0,
            "cleanup_errors": int((cleanup_exec_df["status"] == "error").sum()) if not cleanup_exec_df.empty else 0,
            "cleanup_remaining": int(len(cleanup_df_after)),
        },
        "top_promoted": top_promoted,
        "outputs": {
            "finance_principles_report_csv": str(bt_csv),
            "ml_principles_report_csv": str(ml_csv),
            "promotion_gate_report_csv": str(gate_csv),
            "monitoring_metrics_report_csv": str(mon_csv),
            "folder_cleanup_plan_csv": str(cp_csv),
            "folder_cleanup_execution_report_csv": str(cx_csv),
            "folder_cleanup_remaining_plan_csv": str(cq_csv),
            "quality_metrics_report_csv": str(q_csv),
        },
    }

    summary_json = out_dir / "master_brain_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = out_dir / "master_brain_summary.md"
    lines: list[str] = []
    lines.append("# Master Brain Summary")
    lines.append("")
    lines.append(f"- Backtests scored: **{summary['counts']['backtests_scored']}**")
    lines.append(f"- ML runs scored: **{summary['counts']['ml_runs_scored']}**")
    lines.append(f"- Promoted runs: **{summary['counts']['promoted_runs']}**")
    lines.append(f"- Planned cleanup moves: **{summary['counts']['cleanup_moves_planned']}**")
    lines.append(f"- Cleanup actions executed: **{summary['counts']['cleanup_actions_executed']}**")
    lines.append(f"- Cleanup errors: **{summary['counts']['cleanup_errors']}**")
    lines.append(f"- Cleanup remaining: **{summary['counts']['cleanup_remaining']}**")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- `{bt_csv}`")
    lines.append(f"- `{ml_csv}`")
    lines.append(f"- `{gate_csv}`")
    lines.append(f"- `{mon_csv}`")
    lines.append(f"- `{cp_csv}`")
    lines.append(f"- `{cx_csv}`")
    lines.append(f"- `{cq_csv}`")
    lines.append(f"- `{q_csv}`")
    lines.append(f"- `{summary_json}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {bt_csv}")
    print(f"Wrote: {ml_csv}")
    print(f"Wrote: {gate_csv}")
    print(f"Wrote: {mon_csv}")
    print(f"Wrote: {cp_csv}")
    print(f"Wrote: {cx_csv}")
    print(f"Wrote: {cq_csv}")
    print(f"Wrote: {q_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
