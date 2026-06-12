"""Dissect ORB losses: where does a raw breakout bleed, and what fixes it?

Runs a base 1:1-RR ORB (no filters) to get a large trade sample, engineers
per-trade features, and reports win-rate / profit-factor / expectancy broken
down by side, entry time, day of week, trend alignment, and opening-range size.
The bottom section stacks the favourable conditions to show the lift.

    python examples/orb_analysis.py --csv nq_5m.csv

Findings on NQ (2022-2026) that drive the recommended config:
  - shorts barely break even (NQ long bias); against-trend trades lose
  - Mondays underperform (weekend-gap digestion)
  - mid-volatility days chop; the widest opening ranges pay best
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import NQ, ORBParams, backtest_orb
from qstack.research.orb import Instrument, compute_features, day_groups

INSTRUMENTS = {"NQ": NQ}


def _pf(sub):
    win, loss = sub.loc[sub.pnl > 0, "pnl"].sum(), -sub.loc[sub.pnl < 0, "pnl"].sum()
    return win / loss if loss > 0 else np.inf


def build_features(df: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    g = day_groups(df)
    feat = compute_features(groups=g, or_minutes=30)
    daily = pd.DataFrame({"date": feat.dates, "or_range": feat.or_range, "dclose": feat.daily_close})
    daily["sma50"] = daily["dclose"].shift(1).rolling(50).mean()
    daily["prev_close"] = daily["dclose"].shift(1)
    daily["or_med20"] = pd.Series(feat.or_range).shift(1).rolling(20).median().values
    daily = daily.set_index("date")

    t = backtest_orb(df, ORBParams(or_minutes=30, stop_mult=1.0, target_mult=1.0, direction="both"), inst).trades.copy()
    et = pd.to_datetime(t["entry_time"])
    t["entry_min"] = et.dt.hour * 60 + et.dt.minute - (9 * 60 + 30)
    t["dow"] = et.dt.day_name().str[:3]
    t["win"] = t["pnl"] > 0
    t["or_rel"] = t["or_range"] / t["entry"]
    t = t.join(daily[["prev_close", "sma50", "or_med20"]], on="date")
    t["trend_up"] = t["prev_close"] > t["sma50"]
    t["with_trend"] = ((t["side"] == "long") & t["trend_up"]) | ((t["side"] == "short") & ~t["trend_up"])
    t["or_big"] = t["or_range"] > t["or_med20"]
    t["emin_bucket"] = pd.cut(t["entry_min"], [-1, 15, 30, 60, 120, 400],
                              labels=["0-15", "15-30", "30-60", "60-120", "120+"])
    return t


def summary(t: pd.DataFrame, by: str) -> str:
    rows = []
    for k, sub in t.groupby(by, observed=True):
        rows.append({by: k, "n": len(sub), "win%": round(sub.win.mean() * 100, 1),
                     "PF": round(_pf(sub), 2), "avg$": round(sub.pnl.mean()),
                     "net$": round(sub.pnl.sum())})
    return pd.DataFrame(rows).to_string(index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--instrument", default="NQ", choices=sorted(INSTRUMENTS))
    args = p.parse_args(argv)

    t = build_features(load_ohlcv_csv(args.csv), INSTRUMENTS[args.instrument])
    print(f"OVERALL (1:1 RR, no filters): n={len(t)} win={t.win.mean()*100:.1f}% "
          f"PF={_pf(t):.2f} net=${t.pnl.sum():,.0f}")
    for label, col in [("SIDE", "side"), ("ENTRY TIME", "emin_bucket"),
                       ("DAY OF WEEK", "dow"), ("WITH vs AGAINST TREND", "with_trend"),
                       ("EXIT REASON", "reason")]:
        print(f"\n--- by {label} ---")
        print(summary(t, col))

    t["or_q"] = pd.qcut(t["or_rel"], 5, labels=["Q1 tight", "Q2", "Q3", "Q4", "Q5 wide"])
    print("\n--- by OPENING-RANGE size quintile ---")
    print(summary(t, "or_q"))

    print("\n=== stacking the favourable conditions (1:1 RR) ===")
    for desc, mask in [
        ("all trades", pd.Series(True, index=t.index)),
        ("with trend", t.with_trend),
        ("with trend & entry<=30m", t.with_trend & (t.entry_min <= 30)),
        ("with trend & entry<=30m & wide OR", t.with_trend & (t.entry_min <= 30) & t.or_big),
    ]:
        s = t[mask]
        print(f"  {desc:38s} n={len(s):4d}  win={s.win.mean()*100:5.1f}%  "
              f"PF={_pf(s):4.2f}  avg=${s.pnl.mean():6.0f}")


if __name__ == "__main__":
    main()
