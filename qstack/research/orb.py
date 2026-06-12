"""Opening Range Breakout (ORB) — an intraday futures day-trading engine.

The opening range is the high/low of the first ``or_minutes`` of the Regular
Trading Hours session (09:30 ET for index futures). After that window closes we
trade the first breakout of the range, place a protective stop and an optional
profit target sized as multiples of the range, and flat the position by the
session close (a pure day-trade — nothing held overnight).

Optional filters, all designed to keep a raw ORB from bleeding in chop:
- trend filter  (``trend_ma``):      only take breakouts in the direction of the
                                     daily trend (prior close vs an N-day SMA).
- volatility    (``vol_min_frac``):  skip days whose opening range is small
                                     relative to its recent median (no follow-through).
- entry cutoff  (``entry_cutoff``):  take no new entries after a time of day
                                     (most ORB edge is in the first hours).

Everything is sized in real futures economics: points, ticks, point value, plus
commission and slippage, so P&L is in dollars per 1-lot.

Honesty / mechanics:
- Timestamps are treated as bar-OPEN times (a 09:30 bar covers 09:30-09:34:59).
- Stop/target are checked from the bar AFTER entry to avoid intrabar look-ahead.
  If one later bar trades through both, we assume the STOP filled first.
- Gaps: if a bar opens beyond the breakout/stop/target level, the fill is the
  bar open, not the level.
- Trend/vol filters use only PRIOR-day information (shifted), so no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
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

VOL_MEDIAN_LOOKBACK = 20   # days of prior opening ranges used by the vol filter


@dataclass(frozen=True)
class ORBParams:
    or_minutes: int = 30
    direction: str = "both"             # both | long | short
    stop_mult: float = 1.0              # stop distance = stop_mult * OR range
    target_mult: float | None = 2.0     # target distance = target_mult * OR range; None -> EOD exit only
    trend_ma: int = 0                   # 0 = off; else require prior close on the right side of SMA(trend_ma)
    vol_min_frac: float = 0.0           # 0 = off; require OR range >= frac * median(prior ranges)
    entry_cutoff: time = time(16, 0)    # no new entries at/after this ET time
    confirm_close: bool = False         # require the breakout BAR to CLOSE beyond the range; enter next bar open
    breakeven_at: float | None = None   # once price runs this * range in favor, move the stop to entry
    skip_monday: bool = False           # drop Mondays (weekend-gap digestion days bleed)
    session_open: time = time(9, 30)
    session_close: time = time(16, 0)


def trade_stats(trades: pd.DataFrame, n_days: int) -> dict:
    """Performance summary for a set of trades (works on any subset/window)."""
    if trades.empty:
        return {"n_trades": 0, "net_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_trade": 0.0, "expectancy": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                "n_days": n_days, "pct_days_traded": 0.0}
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    equity = pnl.cumsum()
    drawdown = (equity - equity.cummax()).min()
    daily = trades.groupby("date")["pnl"].sum()
    daily = daily.reindex(pd.Index(sorted(set(trades["date"]))), fill_value=0.0)
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    return {
        "n_trades": int(len(trades)),
        "net_pnl": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() != 0 else float("inf"),
        "avg_trade": float(pnl.mean()),
        "expectancy": float(pnl.mean()),
        "max_drawdown": float(drawdown),
        "sharpe": float(sharpe),
        "n_days": n_days,
        "pct_days_traded": float(trades["date"].nunique() / n_days) if n_days else 0.0,
    }


@dataclass
class ORBResult:
    trades: pd.DataFrame
    params: ORBParams
    instrument: Instrument
    n_days: int

    @property
    def stats(self) -> dict:
        return trade_stats(self.trades, self.n_days)

    def equity_curve(self) -> pd.Series:
        if self.trades.empty:
            return pd.Series(dtype=float)
        eq = self.trades.set_index("exit_time")["pnl"].cumsum()
        eq.name = "equity"
        return eq

    def __repr__(self) -> str:
        s = self.stats
        return (f"ORBResult({self.instrument.name} OR{self.params.or_minutes}m "
                f"{self.params.direction} stop={self.params.stop_mult} tgt={self.params.target_mult} "
                f"trend={self.params.trend_ma} vol={self.params.vol_min_frac} | "
                f"trades={s['n_trades']} net=${s['net_pnl']:,.0f} "
                f"PF={s['profit_factor']:.2f} win={s['win_rate']:.1%} "
                f"maxDD=${s['max_drawdown']:,.0f} sharpe={s['sharpe']:.2f})")


def day_groups(df: pd.DataFrame) -> list:
    """Split an intraday frame into ``(date, day_df)`` pairs (computed once)."""
    if df.index.tz is None:
        raise ValueError("df index must be tz-aware (US/Eastern)")
    df = df.sort_index()
    return list(df.groupby(df.index.date, sort=True))


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


class _DayFeatures:
    """Per-day arrays + opening-range stats for a fixed ``or_minutes``.

    Depends only on ``or_minutes`` and the session window, NOT on stop/target/
    trend/vol/cutoff — so a parameter sweep computes this once and reruns the
    cheap simulation for every other combination.
    """

    __slots__ = ("dates", "ts", "o", "h", "lo", "c", "m", "or_high", "or_low",
                 "or_range", "daily_close", "n_days")

    def __init__(self, groups, or_minutes, open_min, close_min):
        or_close_min = open_min + or_minutes
        self.n_days = len(groups)
        self.dates, self.ts = [], []
        self.o, self.h, self.lo, self.c, self.m = [], [], [], [], []
        self.or_high, self.or_low, self.or_range, self.daily_close = [], [], [], []
        for d, day in groups:
            idx = day.index
            mins = (idx.hour * 60 + idx.minute).to_numpy()
            rth = (mins >= open_min) & (mins < close_min)
            if rth.sum() < 3 or mins[rth][0] != open_min:
                continue
            m = mins[rth]
            h = day["high"].to_numpy()[rth]
            lo = day["low"].to_numpy()[rth]
            or_sel = m < or_close_min
            if not or_sel.any() or not (~or_sel).any():
                continue
            oh, ol = h[or_sel].max(), lo[or_sel].min()
            if oh - ol <= 0:
                continue
            self.dates.append(d)
            self.ts.append(idx[rth])
            self.o.append(day["open"].to_numpy()[rth])
            self.h.append(h)
            self.lo.append(lo)
            self.c.append(day["close"].to_numpy()[rth])
            self.m.append(m)
            self.or_high.append(oh)
            self.or_low.append(ol)
            self.or_range.append(oh - ol)
            self.daily_close.append(day["close"].to_numpy()[rth][-1])


def compute_features(df=None, or_minutes=30, groups=None,
                     session_open=time(9, 30), session_close=time(16, 0)) -> _DayFeatures:
    if groups is None:
        groups = day_groups(df)
    return _DayFeatures(groups, or_minutes, _minutes(session_open), _minutes(session_close))


def _filter_masks(feat: _DayFeatures, params: ORBParams):
    """Build per-day trend-direction and volatility allow masks (prior-day only)."""
    n = len(feat.dates)
    if params.trend_ma > 0:
        pc = pd.Series(feat.daily_close).shift(1)
        sma = pc.rolling(params.trend_ma).mean()
        long_ok = (pc > sma).to_numpy()
        short_ok = (pc < sma).to_numpy()
    else:
        long_ok = np.ones(n, dtype=bool)
        short_ok = np.ones(n, dtype=bool)

    if params.vol_min_frac > 0:
        orr = pd.Series(feat.or_range)
        med = orr.shift(1).rolling(VOL_MEDIAN_LOOKBACK).median()
        vol_ok = (orr >= params.vol_min_frac * med).to_numpy()
        vol_ok = np.where(np.isnan(med.to_numpy()), False, vol_ok)
    else:
        vol_ok = np.ones(n, dtype=bool)
    return long_ok, short_ok, vol_ok


def _simulate(feat: _DayFeatures, params: ORBParams, inst: Instrument) -> pd.DataFrame:
    slip = inst.slippage_pts
    long_ok, short_ok, vol_ok = _filter_masks(feat, params)
    cutoff_min = _minutes(params.entry_cutoff)
    allow_long = params.direction in ("both", "long")
    allow_short = params.direction in ("both", "short")

    trades = []
    for k in range(len(feat.dates)):
        if not vol_ok[k]:
            continue
        if params.skip_monday and pd.Timestamp(feat.dates[k]).weekday() == 0:
            continue
        al = allow_long and long_ok[k]
        ash = allow_short and short_ok[k]
        if not (al or ash):
            continue

        o, h, lo, c, m, ts = feat.o[k], feat.h[k], feat.lo[k], feat.c[k], feat.m[k], feat.ts[k]
        or_high, or_low, or_range = feat.or_high[k], feat.or_low[k], feat.or_range[k]
        or_close_min = _minutes(params.session_open) + params.or_minutes

        # --- first breakout, entries only before the cutoff ---
        # With confirm_close, the breakout BAR must close beyond the range and we
        # enter at the next bar's open (filters wick-through false breakouts).
        side = entry = entry_i = None
        for i in range(len(m)):
            if m[i] < or_close_min or m[i] >= cutoff_min:
                continue
            long_hit = al and (c[i] > or_high if params.confirm_close else h[i] >= or_high)
            short_hit = ash and (c[i] < or_low if params.confirm_close else lo[i] <= or_low)
            if long_hit and short_hit:
                long_hit, short_hit = c[i] >= o[i], c[i] < o[i]
            if not (long_hit or short_hit):
                continue
            if params.confirm_close:
                if i + 1 >= len(m):
                    break
                entry_i = i + 1
                entry = (o[entry_i] + slip) if long_hit else (o[entry_i] - slip)
            else:
                entry_i = i
                entry = (max(or_high, o[i]) + slip) if long_hit else (min(or_low, o[i]) - slip)
            side = "long" if long_hit else "short"
            break
        if side is None:
            continue

        stop_dist = params.stop_mult * or_range
        tgt_dist = None if params.target_mult is None else params.target_mult * or_range
        be_dist = None if params.breakeven_at is None else params.breakeven_at * or_range
        if side == "long":
            stop_px = entry - stop_dist
            tgt_px = None if tgt_dist is None else entry + tgt_dist
        else:
            stop_px = entry + stop_dist
            tgt_px = None if tgt_dist is None else entry - tgt_dist

        exit_px = exit_time = reason = None
        be_done = be_dist is None
        for i in range(entry_i + 1, len(m)):
            if side == "long":
                if lo[i] <= stop_px:
                    exit_px, reason = min(stop_px, o[i]) - slip, ("breakeven" if be_done and stop_px >= entry else "stop")
                elif tgt_px is not None and h[i] >= tgt_px:
                    exit_px, reason = max(tgt_px, o[i]) - slip, "target"
                elif not be_done and h[i] >= entry + be_dist:
                    stop_px, be_done = entry, True  # lock to breakeven for later bars
            else:
                if h[i] >= stop_px:
                    exit_px, reason = max(stop_px, o[i]) + slip, ("breakeven" if be_done and stop_px <= entry else "stop")
                elif tgt_px is not None and lo[i] <= tgt_px:
                    exit_px, reason = min(tgt_px, o[i]) + slip, "target"
                elif not be_done and lo[i] <= entry - be_dist:
                    stop_px, be_done = entry, True
            if exit_px is not None:
                exit_time = ts[i]
                break

        if exit_px is None:  # flat at session close
            exit_time = ts[-1]
            exit_px = (c[-1] - slip) if side == "long" else (c[-1] + slip)
            reason = "eod"

        gross_pts = (exit_px - entry) if side == "long" else (entry - exit_px)
        pnl = gross_pts * inst.point_value - inst.commission_rt
        trades.append({
            "date": feat.dates[k], "side": side, "entry_time": ts[entry_i], "entry": entry,
            "exit_time": exit_time, "exit": exit_px, "reason": reason,
            "or_range": or_range, "points": gross_pts, "pnl": pnl,
        })

    cols = ["date", "side", "entry_time", "entry", "exit_time", "exit",
            "reason", "or_range", "points", "pnl"]
    return pd.DataFrame(trades, columns=cols)


def backtest_orb(df: pd.DataFrame, params: ORBParams, instrument: Instrument,
                 groups: list | None = None, features: _DayFeatures | None = None) -> ORBResult:
    """Backtest the ORB strategy on ET-indexed intraday OHLCV.

    Pass ``features`` (from :func:`compute_features` for the same ``or_minutes``)
    to skip the per-day preprocessing during a sweep.
    """
    if features is None:
        features = compute_features(df, params.or_minutes, groups,
                                    params.session_open, params.session_close)
    trades = _simulate(features, params, instrument)
    return ORBResult(trades, params, instrument, features.n_days)


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame      # one row per test window: chosen params + train/test stats
    oos_trades: pd.DataFrame   # concatenated out-of-sample trades across all windows
    n_oos_days: int

    @property
    def stats(self) -> dict:
        return trade_stats(self.oos_trades, self.n_oos_days)

    def equity_curve(self) -> pd.Series:
        if self.oos_trades.empty:
            return pd.Series(dtype=float)
        eq = self.oos_trades.sort_values("exit_time").set_index("exit_time")["pnl"].cumsum()
        eq.name = "equity"
        return eq


def walk_forward(df: pd.DataFrame, grid: list[ORBParams], instrument: Instrument,
                 train_days: int = 252, test_days: int = 63, metric: str = "sharpe",
                 min_train_trades: int = 30, min_pf: float = 1.0,
                 trades_full: list | None = None,
                 all_dates: list | None = None) -> WalkForwardResult:
    """Rolling walk-forward: re-optimize on a trailing window, trade the next.

    For every test window the best config (by ``metric``, requiring a positive
    in-sample profit factor) is chosen on the preceding ``train_days`` and then
    applied — untouched — to the next ``test_days``. The concatenation of those
    test windows is a fully out-of-sample track record.

    Because the trend/vol filters use only prior-day data, each config is
    simulated once over the full history and then sliced by trade date into
    train/test windows — exact and fast. Pass ``trades_full``/``all_dates`` to
    reuse a full-history simulation already computed by the caller.
    """
    if trades_full is None or all_dates is None:
        groups = day_groups(df)
        all_dates = [d for d, _ in groups]
        feats = {om: compute_features(groups=groups, or_minutes=om)
                 for om in {p.or_minutes for p in grid}}
        trades_full = [_simulate(feats[p.or_minutes], p, instrument) for p in grid]

    rows, oos_frames = [], []
    i = train_days
    while i < len(all_dates):
        train_dates = set(all_dates[i - train_days:i])
        test_dates = all_dates[i:i + test_days]
        if not test_dates:
            break
        test_set = set(test_dates)

        best_j, best_val = None, -np.inf
        for j, p in enumerate(grid):
            sub = trades_full[j][trades_full[j]["date"].isin(train_dates)]
            if len(sub) < min_train_trades:
                continue
            st = trade_stats(sub, len(train_dates))
            if st["profit_factor"] >= min_pf and st[metric] > best_val:
                best_val, best_j = st[metric], j

        if best_j is not None:
            p = grid[best_j]
            test_sub = trades_full[best_j][trades_full[best_j]["date"].isin(test_set)]
            oos_frames.append(test_sub)
            ts = trade_stats(test_sub, len(test_dates))
            rows.append({
                "test_start": test_dates[0], "test_end": test_dates[-1],
                "or_minutes": p.or_minutes, "direction": p.direction,
                "stop_mult": p.stop_mult, "target_mult": p.target_mult,
                "trend_ma": p.trend_ma, "vol_min_frac": p.vol_min_frac,
                "cutoff": p.entry_cutoff.strftime("%H:%M"),
                "train_" + metric: round(best_val, 3),
                "test_trades": ts["n_trades"], "test_net": ts["net_pnl"],
                "test_pf": round(ts["profit_factor"], 3), "test_sharpe": round(ts["sharpe"], 3),
            })
        i += test_days

    oos = pd.concat(oos_frames).sort_values("exit_time") if oos_frames else pd.DataFrame()
    n_oos_days = sum(len(all_dates[k:k + test_days]) for k in range(train_days, len(all_dates), test_days))
    return WalkForwardResult(pd.DataFrame(rows), oos, n_oos_days)
