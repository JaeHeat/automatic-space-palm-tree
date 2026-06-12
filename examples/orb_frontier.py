"""Map the win-rate vs profit-factor frontier for ORB on a futures CSV.

Sweeps reward:risk and the filters, then plots every config as a point in
(win rate, profit factor) space. The point of the chart: the two goals
"win > 60%" and "PF > 1.5" sit in opposite corners of the frontier — you buy a
higher hit rate with a tighter target, which lowers PF, and vice-versa.

    python examples/orb_frontier.py --csv nq_5m.csv --plot orb_frontier.png

Also prints three robust operating points (max-win, max-PF, balanced), where
"robust" means the profit factor holds up on BOTH the train and test halves.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import time

import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import NQ, ORBParams, trade_stats
from qstack.research.orb import Instrument, compute_features, day_groups, _simulate

INSTRUMENTS = {"NQ": NQ}

AXES = {
    "or_minutes": [30, 60],
    "direction": ["both", "long"],
    "stop_mult": [0.75, 1.0, 1.25, 1.5],
    "target_mult": [0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0],
    "trend_ma": [0, 50],
    "vol_min_frac": [0.0, 1.0],
    "entry_cutoff": [time(11, 30)],
    "skip_monday": [True],
}


def sweep(df, inst, split):
    g = day_groups(df)
    all_dates = [d for d, _ in g]
    sd = pd.Timestamp(split).date()
    train = {d for d in all_dates if d < sd}
    test = {d for d in all_dates if d >= sd}
    feats = {om: compute_features(groups=g, or_minutes=om) for om in AXES["or_minutes"]}

    keys = list(AXES)
    rows = []
    for combo in itertools.product(*(AXES[k] for k in keys)):
        p = dict(zip(keys, combo))
        par = ORBParams(**p)
        t = _simulate(feats[p["or_minutes"]], par, inst)
        s = trade_stats(t, len(all_dates))
        if s["n_trades"] < 120:
            continue
        st = trade_stats(t[t["date"].isin(train)], len(train))
        se = trade_stats(t[t["date"].isin(test)], len(test))
        rows.append({"par": par, "rr": p["target_mult"] / p["stop_mult"],
                     "win": s["win_rate"] * 100, "pf": s["profit_factor"],
                     "net": s["net_pnl"], "robust_pf": min(st["profit_factor"], se["profit_factor"]),
                     "tr": st, "te": se, "full": s})
    return pd.DataFrame(rows)


def _spec(par: ORBParams) -> str:
    return (f"OR{par.or_minutes}m {par.direction} stop{par.stop_mult}x tgt{par.target_mult}x "
            f"(RR {par.target_mult/par.stop_mult:.2f}:1) trend{par.trend_ma} vol{par.vol_min_frac} "
            f"cutoff{par.entry_cutoff.strftime('%H:%M')}"
            + (" skipMon" if par.skip_monday else ""))


def _print_point(label, r):
    print(f"\n{label}\n  {_spec(r['par'])}")
    for w, lab in [("full", "full 2022-26"), ("tr", "in-sample"), ("te", "out-of-sample")]:
        s = r[w]
        print(f"    {lab:14s} N={s['n_trades']:4d} win={s['win_rate']*100:5.1f}% "
              f"PF={s['profit_factor']:.2f} net=${s['net_pnl']:>9,.0f} maxDD=${s['max_drawdown']:>9,.0f}")


def plot(R, inst, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(R["win"], R["pf"].clip(upper=3), c=R["rr"], cmap="viridis", s=28, alpha=0.8)
    fig.colorbar(sc, label="reward:risk (target / stop)")
    ax.axvline(60, color="red", ls="--", lw=1)
    ax.axhline(1.5, color="red", ls="--", lw=1)
    ax.text(60.3, 2.9, "TARGET: win>60% & PF>1.5\n(empty — unreachable)",
            color="darkred", fontsize=10, va="top")
    ax.set_xlabel("win rate (%) — full sample")
    ax.set_ylabel("profit factor (capped at 3)")
    ax.set_title(f"qstack ORB {inst.name} — win-rate vs profit-factor frontier (each dot = one config)")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nsaved frontier chart -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--instrument", default="NQ", choices=sorted(INSTRUMENTS))
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    inst = INSTRUMENTS[args.instrument]
    R = sweep(load_ohlcv_csv(args.csv), inst, args.split)
    box = ((R["win"] > 60) & (R["pf"] > 1.5)).sum()
    print(f"{len(R)} configs.  max win={R['win'].max():.1f}%  max PF={R['pf'].max():.2f}  "
          f"configs meeting win>60 AND PF>1.5: {box}")

    rob = R[R["robust_pf"] > 1.2]
    _print_point("A) MAX WIN-RATE (robust)", rob.loc[rob["win"].idxmax()])
    _print_point("B) MAX PROFIT-FACTOR (robust)", rob.loc[rob["pf"].idxmax()])
    bal = R[R["win"] >= 58]
    _print_point("C) BALANCED (win>=58, best robust PF)", bal.loc[bal["robust_pf"].idxmax()])

    if args.plot:
        plot(R, inst, args.plot)


if __name__ == "__main__":
    main()
