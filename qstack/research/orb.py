"""Opening Range Breakout (ORB) — an intraday futures day-trading engine.

The opening range is the high/low of the first ``or_minutes`` of the Regular
Trading Hours session (09:30 ET for index futures). After that window closes we
trade the first breakout of the range, place a protective stop and an optional
profit target sized as multiples of the range, and flat the position by the
session close (a pure day-trade — nothing held overnight).

Everything is sized in real futures economics: points, ticks, point value, plus
commission and slippage, so the P&L is in dollars per 1-lot.

Design notes / honesty:
- Timestamps are treated as bar-OPEN times, so a 09:30 bar covers 09:30-09:34:59.
- Stop/target are checked from the bar AFTER entry to avoid intrabar look-ahead
  on the entry bar. If a single later bar trades through both the stop and the
  target, we assume the STOP filled first (conservative).
- Gaps are handled: if a bar opens beyond the breakout/stop level, the fill is
  the bar open, not the level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Instrument:
    """Contract economics for one futures market."""
    name: str
    tick_size: float
    point_value: float          # dollars per 1.0 point per contract
    commission_rt: float = 4.0  # round-trip commission, dollars
    slippage_ticks: float = 1.0 # slippage per fill, in ticks

    @property
    def slippage_pts(self) -> float:
        return self.slippage_ticks * self.tick_size


# E-mini index futures.
NQ = Instrument("NQ", tick_size=0.25, point_value=20.0)
ES = Instrument("ES", tick_size=0.25, point_value=50.0)


@dataclass(frozen=True)
class ORBParams:
    or_minutes: int = 30
    direction: str = "both"            # both | long | short
    stop_mult: float = 1.0             # stop distance = stop_mult * OR range
    target_mult: float | None = 2.0    # target distance = target_mult * OR range; None -> EOD exit only
    session_open: time = time(9, 30)
    session_close: time = time(16, 0)


@dataclass
class ORBResult:
    trades: pd.DataFrame
    params: ORBParams
    instrument: Instrument
    n_days: int

    @property
    def stats(self) -> dict:
        t = self.trades
        if t.empty:
            return {"n_trades": 0, "net_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                    "avg_trade": 0.0, "expectancy": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                    "n_days": self.n_days, "pct_days_traded": 0.0}
        pnl = t["pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        equity = pnl.cumsum()
        drawdown = (equity - equity.cummax()).min()
        # Sharpe on per-day P&L (days without a trade count as 0).
        daily = t.groupby("date")["pnl"].sum()
        daily = daily.reindex(pd.Index(sorted(set(t["date"]))), fill_value=0.0)
        sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
        return {
            "n_trades": int(len(t)),
            "net_pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() != 0 else float("inf"),
            "avg_trade": float(pnl.mean()),
            "expectancy": float(pnl.mean()),
            "max_drawdown": float(drawdown),
            "sharpe": float(sharpe),
            "n_days": self.n_days,
            "pct_days_traded": float(t["date"].nunique() / self.n_days) if self.n_days else 0.0,
        }

    def equity_curve(self) -> pd.Series:
        if self.trades.empty:
            return pd.Series(dtype=float)
        eq = self.trades.set_index("exit_time")["pnl"].cumsum()
        eq.name = "equity"
        return eq

    def __repr__(self) -> str:
        s = self.stats
        return (f"ORBResult({self.instrument.name} OR{self.params.or_minutes}m "
                f"{self.params.direction} stop={self.params.stop_mult} tgt={self.params.target_mult} | "
                f"trades={s['n_trades']} net=${s['net_pnl']:,.0f} "
                f"PF={s['profit_factor']:.2f} win={s['win_rate']:.1%} "
                f"maxDD=${s['max_drawdown']:,.0f} sharpe={s['sharpe']:.2f})")


def day_groups(df: pd.DataFrame) -> list:
    """Split an intraday frame into ``(date, day_df)`` pairs.

    Computing ``df.index.date`` and grouping is the expensive part of a backtest,
    so a parameter sweep should call this once and pass the result into
    :func:`backtest_orb` via ``groups=`` to avoid repeating it per configuration.
    """
    if df.index.tz is None:
        raise ValueError("df index must be tz-aware (US/Eastern)")
    return list(df.sort_index().groupby(df.sort_index().index.date, sort=True))


def backtest_orb(df: pd.DataFrame, params: ORBParams, instrument: Instrument,
                 groups: list | None = None) -> ORBResult:
    """Backtest the ORB strategy on ET-indexed intraday OHLCV.

    Args:
        df:         OHLCV with a tz-aware DatetimeIndex in US/Eastern, columns
                    open/high/low/close. Ignored if ``groups`` is supplied.
        params:     ORB configuration.
        instrument: contract economics (NQ, ES, or custom).
        groups:     optional precomputed output of :func:`day_groups` to reuse
                    across a parameter sweep.
    """
    if groups is None:
        groups = day_groups(df)
    slip = instrument.slippage_pts

    open_min = params.session_open.hour * 60 + params.session_open.minute
    close_min = params.session_close.hour * 60 + params.session_close.minute
    or_close_min = open_min + params.or_minutes
    allow_long = params.direction in ("both", "long")
    allow_short = params.direction in ("both", "short")

    trades = []
    for d, day in groups:
        idx = day.index
        mins = (idx.hour * 60 + idx.minute).to_numpy()
        rth_mask = (mins >= open_min) & (mins < close_min)
        if rth_mask.sum() < 3 or mins[rth_mask][0] != open_min:
            continue  # need a real 09:30 open and some session

        ts = idx[rth_mask]
        m = mins[rth_mask]
        o = day["open"].to_numpy()[rth_mask]
        h = day["high"].to_numpy()[rth_mask]
        lo = day["low"].to_numpy()[rth_mask]
        c = day["close"].to_numpy()[rth_mask]

        or_sel = m < or_close_min
        rest_sel = ~or_sel
        if not or_sel.any() or not rest_sel.any():
            continue
        or_high, or_low = h[or_sel].max(), lo[or_sel].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        rest_i = [i for i in range(len(m)) if rest_sel[i]]

        # --- find the first breakout ---
        side = entry = entry_i = None
        for i in rest_i:
            long_ok = allow_long and h[i] >= or_high
            short_ok = allow_short and lo[i] <= or_low
            if long_ok and short_ok:
                long_ok, short_ok = c[i] >= o[i], c[i] < o[i]
            if long_ok:
                side, entry, entry_i = "long", max(or_high, o[i]) + slip, i
                break
            if short_ok:
                side, entry, entry_i = "short", min(or_low, o[i]) - slip, i
                break
        if side is None:
            continue
        entry_time = ts[entry_i]

        # --- manage the trade from the bar AFTER entry ---
        stop_dist = params.stop_mult * or_range
        tgt_dist = None if params.target_mult is None else params.target_mult * or_range
        if side == "long":
            stop_px = entry - stop_dist
            tgt_px = None if tgt_dist is None else entry + tgt_dist
        else:
            stop_px = entry + stop_dist
            tgt_px = None if tgt_dist is None else entry - tgt_dist

        exit_px = exit_time = reason = None
        for i in range(entry_i + 1, len(m)):
            if side == "long":
                if lo[i] <= stop_px:
                    exit_px, reason = min(stop_px, o[i]) - slip, "stop"
                elif tgt_px is not None and h[i] >= tgt_px:
                    exit_px, reason = max(tgt_px, o[i]) - slip, "target"
            else:
                if h[i] >= stop_px:
                    exit_px, reason = max(stop_px, o[i]) + slip, "stop"
                elif tgt_px is not None and lo[i] <= tgt_px:
                    exit_px, reason = min(tgt_px, o[i]) + slip, "target"
            if exit_px is not None:
                exit_time = ts[i]
                break

        if exit_px is None:  # neither stop nor target hit -> flat at session close
            exit_time = ts[-1]
            exit_px = (c[-1] - slip) if side == "long" else (c[-1] + slip)
            reason = "eod"

        gross_pts = (exit_px - entry) if side == "long" else (entry - exit_px)
        pnl = gross_pts * instrument.point_value - instrument.commission_rt
        trades.append({
            "date": d, "side": side, "entry_time": entry_time, "entry": entry,
            "exit_time": exit_time, "exit": exit_px, "reason": reason,
            "or_range": or_range, "points": gross_pts, "pnl": pnl,
        })

    cols = ["date", "side", "entry_time", "entry", "exit_time", "exit",
            "reason", "or_range", "points", "pnl"]
    return ORBResult(pd.DataFrame(trades, columns=cols), params, instrument, len(groups))
