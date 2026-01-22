"""Lightweight strategy backtesting utilities.

This module intentionally implements a pragmatic, research-oriented simulator:
- Converts model signals into positions (long/short/flat)
- Applies a 1-bar execution delay to avoid lookahead
- Computes returns on close-to-close (bar returns)
- Applies simple proportional transaction costs/slippage on position changes

It does NOT model order book fills, partial fills, spreads by venue, funding,
margin, borrow, or market impact.

Research/education only; not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    mode: str = "long_short"  # long_short | long_only
    execution_delay: int = 1  # bars
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    initial_equity: float = 1.0
    # If set, scale position magnitude so realized vol ~= vol_target (annualized).
    vol_target: float | None = None
    vol_lookback: int = 20
    max_leverage: float = 3.0


def simulate_ohlc(
    ohlc: pd.DataFrame,
    signal: pd.Series,
    *,
    cfg: BacktestConfig,
    mark: str = "open",
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    trailing_atr_mult: float | None = None,
    atr_window: int = 14,
    stop_priority: str = "conservative",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a single-asset strategy using OHLC bars.

    This is the "next realism step" vs a pure bar-return model:
    - Position changes happen at the bar's mark price (usually next bar Open)
    - Optional stop-loss / take-profit checked intrabar using High/Low
    - Trade list includes explicit entry/exit prices and exit reasons

    Notes:
    - Signals are assumed to be generated at the end of a bar and executed with
      `cfg.execution_delay` bars of delay at the chosen `mark` price.
    - When a stop/TP triggers within a bar, we assume fill at the stop/TP price.
    """
    if ohlc is None or ohlc.empty:
        raise ValueError("ohlc is empty")

    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(set(ohlc.columns)):
        missing = sorted(need - set(ohlc.columns))
        raise ValueError(f"ohlc missing required columns: {missing}")

    df = ohlc.copy()
    for c in ("Open", "High", "Low", "Close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    if df.empty or len(df) < 3:
        raise ValueError("ohlc has too few rows after cleaning")

    mark = (mark or "open").strip().lower()
    if mark not in {"open", "close"}:
        raise ValueError("mark must be 'open' or 'close'")

    # Use open-to-open (or close-to-close) intervals.
    px = df["Open"] if mark == "open" else df["Close"]

    # Align signals to bar index.
    sig = pd.to_numeric(signal, errors="coerce").reindex(df.index).fillna(0.0)
    sig = sig.clip(-1, 1).round(0)
    if (cfg.mode or "long_short").strip().lower() in {"long", "long_only", "long-only"}:
        sig = (sig > 0).astype(float)

    delay = int(cfg.execution_delay)
    if delay < 0:
        delay = 0
    desired = sig.shift(delay).fillna(0.0)

    cost_rate = (float(cfg.cost_bps) + float(cfg.slippage_bps)) / 10000.0
    stop_priority = (stop_priority or "conservative").strip().lower()
    if stop_priority not in {"conservative", "stop", "take_profit", "tp", "aggressive"}:
        raise ValueError("stop_priority must be conservative|stop|take_profit")

    sl = float(stop_loss) if stop_loss is not None else None
    tp = float(take_profit) if take_profit is not None else None
    ts = float(trailing_stop) if trailing_stop is not None else None
    atr_mult = float(trailing_atr_mult) if trailing_atr_mult is not None else None
    atr_window = int(atr_window)
    if sl is not None and sl <= 0:
        sl = None
    if tp is not None and tp <= 0:
        tp = None
    if ts is not None and ts <= 0:
        ts = None
    if atr_mult is not None and atr_mult <= 0:
        atr_mult = None
    if atr_window < 2:
        atr_window = 14

    # Precompute ATR from OHLC, but use ATR from the prior completed bar for each interval.
    atr_prev = None
    if atr_mult is not None:
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [
                (df["High"] - df["Low"]).abs(),
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_window, min_periods=atr_window).mean()
        atr_prev = atr.shift(1)

    ann = _annualization_factor_from_index(df.index)
    target_bar = None
    if cfg.vol_target is not None:
        vt = float(cfg.vol_target)
        if ann and ann > 0:
            target_bar = vt / float(np.sqrt(ann))
        else:
            target_bar = vt

    # State
    equity = float(cfg.initial_equity)
    pos = 0.0
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None
    high_water: float | None = None
    low_water: float | None = None

    rows: list[dict] = []
    trades: list[dict] = []

    # Iterate bars i where we have a "next" price for mark-to-mark returns.
    idx = df.index
    for i in range(0, len(df) - 1):
        t = idx[i]
        p0 = float(px.iloc[i])
        p1 = float(px.iloc[i + 1])
        hi = float(df["High"].iloc[i])
        lo = float(df["Low"].iloc[i])

        # Determine desired position magnitude (direction from desired, size from vol targeting).
        desired_dir = float(desired.iloc[i])
        desired_pos = desired_dir
        if target_bar is not None:
            lb = max(5, int(cfg.vol_lookback))
            # Use mark-to-mark returns for realized vol.
            window = px.pct_change(1).iloc[max(0, i - lb + 1) : i + 1]
            rv = float(window.std(ddof=0)) if len(window.dropna()) >= lb else float("nan")
            if np.isfinite(rv) and rv > 0:
                scale = float(target_bar) / rv
                scale = float(np.clip(scale, 0.0, float(cfg.max_leverage)))
            else:
                scale = 0.0
            desired_pos = desired_dir * scale

        # Rebalance at bar mark (open/close) to match desired.
        turnover_rebal = abs(desired_pos - pos)
        cost_rebal = cost_rate * turnover_rebal

        # Handle trade close/open on rebalancing.
        exit_reason = None
        if pos != 0.0 and desired_pos != pos:
            # Close existing trade at p0
            if entry_time is not None and entry_price is not None:
                if pos > 0:
                    tr = p0 / float(entry_price) - 1.0
                    side = "long"
                else:
                    tr = float(entry_price) / p0 - 1.0
                    side = "short"
                trades.append(
                    {
                        "entry_time": str(entry_time),
                        "exit_time": str(t),
                        "side": side,
                        "entry_price": float(entry_price),
                        "exit_price": float(p0),
                        "return": float(tr),
                        "exit_reason": "signal",
                    }
                )
            entry_time = None
            entry_price = None
            high_water = None
            low_water = None
            exit_reason = "signal"

        if desired_pos != 0.0 and pos != desired_pos:
            # Open new trade at p0
            entry_time = t
            entry_price = float(p0)
            high_water = float(p0)
            low_water = float(p0)

        # Apply rebalance cost immediately.
        pos = desired_pos

        # Now simulate the holding interval until next mark, with possible stop/TP intrabar.
        gross_r = 0.0
        extra_cost = 0.0
        stop_hit = False
        stop_exit_price = None
        stop_kind = None

        if pos != 0.0:
            # Check stop/TP relative to the original trade entry price.
            if (sl is not None or tp is not None or ts is not None or atr_mult is not None) and entry_price is not None:
                ep = float(entry_price)
                if pos > 0:
                    stop_price = ep * (1.0 - sl) if sl is not None else None
                    tp_price = ep * (1.0 + tp) if tp is not None else None
                    hit_stop = bool(stop_price is not None and lo <= float(stop_price))
                    hit_tp = bool(tp_price is not None and hi >= float(tp_price))
                else:
                    stop_price = ep * (1.0 + sl) if sl is not None else None
                    tp_price = ep * (1.0 - tp) if tp is not None else None
                    hit_stop = bool(stop_price is not None and hi >= float(stop_price))
                    hit_tp = bool(tp_price is not None and lo <= float(tp_price))

                # Trailing stop (percent), computed conservatively from watermark
                # from prior bars only (we update watermark after stop checks).
                trail_price = None
                hit_trail = False
                trail_kind = None
                if ts is not None:
                    if pos > 0:
                        hw = float(high_water) if high_water is not None else float(ep)
                        trail_price = hw * (1.0 - ts)
                        hit_trail = bool(lo <= float(trail_price))
                        trail_kind = "trailing_stop"
                        # Combine with fixed stop: tighter stop is the higher price.
                        if stop_price is not None:
                            stop_price = max(float(stop_price), float(trail_price))
                        else:
                            stop_price = float(trail_price)
                        hit_stop = bool(lo <= float(stop_price))
                    else:
                        lw = float(low_water) if low_water is not None else float(ep)
                        trail_price = lw * (1.0 + ts)
                        hit_trail = bool(hi >= float(trail_price))
                        trail_kind = "trailing_stop"
                        # Combine with fixed stop: tighter stop is the lower price.
                        if stop_price is not None:
                            stop_price = min(float(stop_price), float(trail_price))
                        else:
                            stop_price = float(trail_price)
                        hit_stop = bool(hi >= float(stop_price))

                # Trailing stop (ATR), also from prior-bar watermark and prior-bar ATR.
                if atr_mult is not None and atr_prev is not None:
                    atr_i = float(atr_prev.iloc[i]) if i < len(atr_prev) else float("nan")
                    if np.isfinite(atr_i) and atr_i > 0:
                        if pos > 0:
                            hw = float(high_water) if high_water is not None else float(ep)
                            atr_trail_price = hw - atr_mult * atr_i
                            # Combine: tighter stop is higher for longs
                            if stop_price is not None:
                                if float(atr_trail_price) > float(stop_price):
                                    trail_price = float(atr_trail_price)
                                    trail_kind = "trailing_atr"
                                stop_price = max(float(stop_price), float(atr_trail_price))
                            else:
                                stop_price = float(atr_trail_price)
                                trail_price = float(atr_trail_price)
                                trail_kind = "trailing_atr"
                            hit_stop = bool(lo <= float(stop_price))
                        else:
                            lw = float(low_water) if low_water is not None else float(ep)
                            atr_trail_price = lw + atr_mult * atr_i
                            # Combine: tighter stop is lower for shorts
                            if stop_price is not None:
                                if float(atr_trail_price) < float(stop_price):
                                    trail_price = float(atr_trail_price)
                                    trail_kind = "trailing_atr"
                                stop_price = min(float(stop_price), float(atr_trail_price))
                            else:
                                stop_price = float(atr_trail_price)
                                trail_price = float(atr_trail_price)
                                trail_kind = "trailing_atr"
                            hit_stop = bool(hi >= float(stop_price))

                if hit_stop or hit_tp:
                    # Decide which fills when both are hit in the same bar.
                    choose = None
                    if hit_stop and hit_tp:
                        if stop_priority in {"take_profit", "tp", "aggressive"}:
                            choose = "tp"
                        else:
                            choose = "stop"
                    elif hit_tp:
                        choose = "tp"
                    else:
                        choose = "stop"

                    if choose == "tp":
                        stop_exit_price = float(tp_price)  # type: ignore[arg-type]
                        stop_kind = "take_profit"
                    else:
                        stop_exit_price = float(stop_price)  # type: ignore[arg-type]
                        # If a trailing stop is binding, label it.
                        if stop_price is not None and trail_price is not None and abs(float(stop_price) - float(trail_price)) <= 1e-9:
                            stop_kind = trail_kind or "trailing_stop"
                        else:
                            stop_kind = "stop_loss"

                    stop_hit = True

            if stop_hit and stop_exit_price is not None:
                # Return for this interval is from p0 to stop_exit_price (flat after).
                if pos > 0:
                    gross_r = pos * (float(stop_exit_price) / p0 - 1.0)
                else:
                    gross_r = (-pos) * (p0 / float(stop_exit_price) - 1.0)

                # Exit cost
                extra_cost = cost_rate * abs(pos)

                # Record trade exit at this bar timestamp
                if entry_time is not None and entry_price is not None:
                    side = "long" if pos > 0 else "short"
                    if pos > 0:
                        tr = float(stop_exit_price) / float(entry_price) - 1.0
                    else:
                        tr = float(entry_price) / float(stop_exit_price) - 1.0
                    trades.append(
                        {
                            "entry_time": str(entry_time),
                            "exit_time": str(t),
                            "side": side,
                            "entry_price": float(entry_price),
                            "exit_price": float(stop_exit_price),
                            "return": float(tr),
                            "exit_reason": stop_kind,
                        }
                    )

                # Flat after stop/TP
                pos = 0.0
                entry_time = None
                entry_price = None
                high_water = None
                low_water = None
                exit_reason = stop_kind
            else:
                # No stop: return is mark-to-mark
                if pos > 0:
                    gross_r = pos * (p1 / p0 - 1.0)
                else:
                    gross_r = (-pos) * (p0 / p1 - 1.0)

                # Update trailing watermark AFTER stop checks (conservative ordering).
                if (ts is not None or atr_mult is not None) and entry_price is not None:
                    if pos > 0:
                        high_water = float(max(float(high_water or entry_price), hi))
                    else:
                        low_water = float(min(float(low_water or entry_price), lo))

        net_r = float(gross_r) - float(cost_rebal) - float(extra_cost)
        equity = equity * (1.0 + net_r)

        rows.append(
            {
                "time": str(t),
                "mark": mark,
                "price": float(p0),
                "price_next": float(p1),
                "signal": float(sig.iloc[i]),
                "desired_pos": float(desired_pos),
                "position": float(desired_pos) if not stop_hit else 0.0,
                "turnover": float(turnover_rebal + (abs(desired_pos) if stop_hit else 0.0)),
                "cost": float(cost_rebal + extra_cost),
                "gross_ret": float(gross_r),
                "strategy_ret": float(net_r),
                "equity": float(equity),
                "event": exit_reason,
            }
        )

    curve = pd.DataFrame(rows)
    curve["time"] = pd.to_datetime(curve["time"], errors="coerce")
    curve = curve.dropna(subset=["time"]).set_index("time").sort_index()

    trades_df = pd.DataFrame(trades)
    return curve, trades_df


def _annualization_factor_from_index(index: pd.Index) -> float | None:
    """Infer an annualization factor in "bars per year" from the datetime span.

    Why this approach:
    - Using median delta can over/under-annualize when there are missing dates,
      holidays, or mixed calendars.
    - Using bars-per-year based on the total span tends to land near ~252 for
      equity daily data, and near ~365 for crypto daily data.
    """
    if index is None or len(index) < 3:
        return None

    if isinstance(index, pd.DatetimeIndex):
        dt_index = index
    else:
        try:
            dt_index = pd.to_datetime(index)  # type: ignore[assignment]
        except Exception:
            return None

    try:
        start = dt_index.min()
        end = dt_index.max()
    except Exception:
        return None

    if start is pd.NaT or end is pd.NaT:
        return None

    span = end - start
    try:
        years = float(span / pd.Timedelta(days=365.25))
    except Exception:
        return None
    if not (years and years > 0):
        return None

    n = int(len(dt_index))
    # bars per year
    return float(n / years)


def compute_equity_curve(
    close: pd.Series,
    signal: pd.Series,
    *,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Compute per-bar strategy returns and equity curve.

    Inputs:
      close: close prices indexed by datetime
      signal: desired position signal in {-1,0,1} indexed by datetime

    Output columns:
      close, ret, signal, position, turnover, cost, strategy_ret, equity
    """
    if close is None or close.empty:
        raise ValueError("close series is empty")

    df = pd.DataFrame({"close": pd.to_numeric(close, errors="coerce")}).dropna()
    sig = pd.to_numeric(signal, errors="coerce").reindex(df.index)
    sig = sig.fillna(0.0)

    # Normalize signal to {-1,0,1} direction
    sig = sig.clip(-1, 1).round(0)

    if (cfg.mode or "long_short").strip().lower() in {"long", "long_only", "long-only"}:
        sig = (sig > 0).astype(float)

    df["ret"] = df["close"].pct_change(1)
    df["signal"] = sig

    delay = int(cfg.execution_delay)
    if delay < 0:
        delay = 0

    # Base (directional) position, delayed to avoid lookahead
    pos_dir = df["signal"].shift(delay).fillna(0.0)

    # Optional volatility targeting: position magnitude scales with 1/realized_vol
    pos = pos_dir.copy()
    if cfg.vol_target is not None:
        lb = int(cfg.vol_lookback)
        lb = max(5, lb)
        ann = _annualization_factor_from_index(df.index)
        # Interpret cfg.vol_target as annualized volatility (e.g., 0.20 = 20% annualized)
        if ann and ann > 0:
            target_bar = float(cfg.vol_target) / float(np.sqrt(ann))
        else:
            target_bar = float(cfg.vol_target)

        realized = df["ret"].rolling(lb, min_periods=lb).std(ddof=0)
        # Avoid division by zero
        scale = target_bar / realized.replace(0.0, np.nan)
        scale = scale.clip(lower=0.0, upper=float(cfg.max_leverage)).fillna(0.0)
        pos = (pos_dir * scale).astype(float)

    df["position"] = pos

    # Turnover is abs change in position; -1->1 is 2 for directional positions.
    pos_prev = df["position"].shift(1).fillna(0.0)
    df["turnover"] = (df["position"] - pos_prev).abs()

    cost_rate = (float(cfg.cost_bps) + float(cfg.slippage_bps)) / 10000.0
    df["cost"] = cost_rate * df["turnover"]

    df["strategy_ret"] = (df["position"] * df["ret"]).fillna(0.0) - df["cost"].fillna(0.0)

    equity = [float(cfg.initial_equity)]
    for r in df["strategy_ret"].iloc[1:]:
        equity.append(equity[-1] * (1.0 + float(r)))
    df["equity"] = pd.Series(equity, index=df.index)

    return df


def max_drawdown(equity: pd.Series) -> float:
    e = pd.to_numeric(equity, errors="coerce").dropna()
    if e.empty:
        return float("nan")
    roll_max = e.cummax()
    dd = e / roll_max - 1.0
    return float(dd.min())


def compute_metrics(curve: pd.DataFrame) -> dict:
    if curve is None or curve.empty:
        raise ValueError("curve is empty")

    if "equity" not in curve.columns:
        raise ValueError("curve is missing 'equity' column")
    eq = pd.to_numeric(curve["equity"], errors="coerce").dropna()

    if "strategy_ret" in curve.columns:
        rets = pd.to_numeric(curve["strategy_ret"], errors="coerce").dropna()
    else:
        rets = pd.Series(dtype=float)

    out: dict = {}
    if eq.empty:
        return out

    out["start"] = str(eq.index.min())
    out["end"] = str(eq.index.max())
    out["n_bars"] = int(len(eq))
    out["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    # CAGR
    dt_days = (eq.index.max() - eq.index.min()).days
    years = dt_days / 365.25 if dt_days and dt_days > 0 else None
    if years and years > 0:
        out["cagr"] = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)

    out["max_drawdown"] = float(max_drawdown(eq))

    ann = _annualization_factor_from_index(eq.index)
    out["ann_factor"] = float(ann) if ann is not None else float("nan")
    if ann and rets.std(ddof=0) and float(rets.std(ddof=0)) > 0:
        out["sharpe"] = float(rets.mean() / rets.std(ddof=0) * np.sqrt(ann))

    out["avg_bar_return"] = float(rets.mean()) if not rets.empty else float("nan")
    out["vol_bar"] = float(rets.std(ddof=0)) if not rets.empty else float("nan")

    # Trade-ish stats
    if "position" in curve.columns:
        pos = pd.to_numeric(curve["position"], errors="coerce").fillna(0.0)
        if "turnover" in curve.columns:
            out["turnover_sum"] = float(pd.to_numeric(curve["turnover"], errors="coerce").fillna(0.0).sum())
        else:
            out["turnover_sum"] = float("nan")
        out["n_position_changes"] = int((pos.diff().fillna(0.0) != 0.0).sum())

    return out


def compute_trade_metrics(trades: pd.DataFrame) -> dict:
    """Compute trade-level statistics from a trades DataFrame.

    Expected columns:
      - return: per-trade return as a fraction (e.g., 0.02 = +2%)

    Optional columns:
      - exit_reason: stop_loss | take_profit | trailing_stop | trailing_atr | signal | ...

    Notes:
      - This is intentionally simple and robust to missing/empty inputs.
      - Trade returns in this repo are already direction-adjusted (long/short handled in trade extraction).
    """
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0,
            "n_wins": 0,
            "n_losses": 0,
            "n_breakeven": 0,
            "win_rate": float("nan"),
            "win_loss_ratio": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "profit_factor": float("nan"),
            "expectancy": float("nan"),
        }

    if "return" not in trades.columns:
        r = pd.Series(dtype=float)
    else:
        r = pd.to_numeric(trades["return"], errors="coerce").dropna()
    if r.empty:
        return {
            "n_trades": int(len(trades)),
            "n_wins": 0,
            "n_losses": 0,
            "n_breakeven": 0,
            "win_rate": float("nan"),
            "win_loss_ratio": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "profit_factor": float("nan"),
            "expectancy": float("nan"),
        }

    wins = r[r > 0]
    losses = r[r < 0]
    be = r[r == 0]

    n_trades = int(len(r))
    n_wins = int(len(wins))
    n_losses = int(len(losses))
    n_be = int(len(be))

    gross_profit = float(wins.sum())
    gross_loss_abs = float((-losses).sum())

    out: dict = {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_breakeven": n_be,
        "win_rate": float(n_wins / n_trades) if n_trades > 0 else float("nan"),
        # Win/loss ratio as counts (not average win / average loss).
        "win_loss_ratio": float(n_wins / n_losses) if n_losses > 0 else float("inf") if n_wins > 0 else float("nan"),
        "avg_win": float(wins.mean()) if n_wins > 0 else float("nan"),
        "avg_loss": float(losses.mean()) if n_losses > 0 else float("nan"),
        "profit_factor": float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else float("inf") if gross_profit > 0 else float("nan"),
        "expectancy": float(r.mean()),
        "median_trade_return": float(r.median()),
    }

    # Optional breakdown by exit reason (when available)
    if "exit_reason" in trades.columns:
        vc = trades["exit_reason"].astype(str).value_counts(dropna=True)
        out["exit_reason_counts"] = {str(k): int(v) for k, v in vc.items()}

    return out


def trades_from_positions(
    close: pd.Series,
    position: pd.Series,
) -> pd.DataFrame:
    """Extract a simple trade list from a position series.

    Assumes position is the held position per bar (already delayed).
    Trade return is computed from entry close to exit close (no costs here).
    """
    px = pd.to_numeric(close, errors="coerce")
    pos = pd.to_numeric(position, errors="coerce").fillna(0.0)
    df = pd.DataFrame({"close": px, "position": pos}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "side", "entry_price", "exit_price", "return"])

    trades: list[dict] = []
    current_side: float = 0.0
    entry_time = None
    entry_price = None

    for t, row in df.iterrows():
        p = float(row["position"])
        c = float(row["close"])

        if current_side == 0.0 and p != 0.0:
            current_side = p
            entry_time = t
            entry_price = c
            continue

        if current_side != 0.0 and p != current_side:
            # close existing
            if entry_time is not None and entry_price is not None:
                side = "long" if current_side > 0 else "short"
                raw_ret = c / float(entry_price) - 1.0
                tr = raw_ret if current_side > 0 else -raw_ret
                trades.append(
                    {
                        "entry_time": str(entry_time),
                        "exit_time": str(t),
                        "side": side,
                        "entry_price": float(entry_price),
                        "exit_price": float(c),
                        "return": float(tr),
                    }
                )

            # open new if p != 0
            if p != 0.0:
                current_side = p
                entry_time = t
                entry_price = c
            else:
                current_side = 0.0
                entry_time = None
                entry_price = None

    return pd.DataFrame(trades)
