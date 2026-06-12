"""Find a robust ORB day-trading strategy on ES/NQ futures — honestly.

Builds an ORB parameter grid that includes a trend filter, a volatility filter,
and an entry-time cutoff, then evaluates it two ways:

  1. SPLIT      optimize on a TRAIN window, report a single held-out TEST window.
  2. WALKFORWARD re-optimize on a rolling trailing window and trade the next
                quarter, repeatedly -- a fully out-of-sample track record.

    python examples/orb_futures.py --csv nq.csv --instrument NQ --split 2025-01-01 --plot orb_nq.png

CSV format: epoch-second `time` column + open/high/low/close[/volume]
(Databento 5m export).
"""

from __future__ import annotations

import argparse
import itertools
from datetime import time

import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import ES, NQ, ORBParams, trade_stats, walk_forward
from qstack.research.orb import Instrument, compute_features, day_groups, _simulate

INSTRUMENTS = {"NQ": NQ, "ES": ES}

# Search grid. The filters (trend / vol / cutoff) are the whole point of this
# pass -- the question is whether they make the edge survive out-of-sample.
GRID_AXES = {
    "or_minutes": [15, 30, 60],
    "stop_mult": [0.5, 1.0, 1.5],
    "target_mult": [2.0, None],
    "trend_ma": [0, 50],
    "vol_min_frac": [0.0, 0.8],
    "entry_cutoff": [time(11, 30), time(16, 0)],
}
MIN_TRAIN_TRADES = 80


def build_grid() -> list[ORBParams]:
    keys = list(GRID_AXES)
    return [ORBParams(direction="both", **dict(zip(keys, combo)))
            for combo in itertools.product(*(GRID_AXES[k] for k in keys))]


def _fmt(p: ORBParams) -> str:
    return (f"OR{p.or_minutes}m stop{p.stop_mult} tgt{p.target_mult} "
            f"trend{p.trend_ma} vol{p.vol_min_frac} cutoff{p.entry_cutoff.strftime('%H:%M')}")


def _print_stats(label, s):
    print(f"    {label:<22} trades={s['n_trades']:>4}  net=${s['net_pnl']:>10,.0f}  "
          f"PF={s['profit_factor']:.2f}  win={s['win_rate']:.1%}  "
          f"maxDD=${s['max_drawdown']:>9,.0f}  sharpe={s['sharpe']:.2f}")


def run(csv, inst: Instrument, split, plot):
    df = load_ohlcv_csv(csv)
    groups = day_groups(df)
    all_dates = [d for d, _ in groups]
    grid = build_grid()
    print(f"\n=== {inst.name} ===  {len(df):,} bars  {all_dates[0]} -> {all_dates[-1]}  "
          f"({len(all_dates)} days, {len(grid)} configs)")

    # Simulate every config once over the full history (filters use prior-day
    # data only, so windows are exact slices of these results).
    feats = {om: compute_features(groups=groups, or_minutes=om)
             for om in {p.or_minutes for p in grid}}
    trades_full = [_simulate(feats[p.or_minutes], p, inst) for p in grid]

    # ---- 1) single train/test split ----
    split_ts = pd.Timestamp(split).date()
    train_dates = {d for d in all_dates if d < split_ts}
    test_dates = {d for d in all_dates if d >= split_ts}

    best_j, best_val = None, float("-inf")
    for j in range(len(grid)):
        sub = trades_full[j][trades_full[j]["date"].isin(train_dates)]
        if len(sub) < MIN_TRAIN_TRADES:
            continue
        st = trade_stats(sub, len(train_dates))
        if st["profit_factor"] > 1.0 and st["sharpe"] > best_val:
            best_val, best_j = st["sharpe"], j

    print(f"\n[1] TRAIN/TEST SPLIT at {split}  (train {len(train_dates)}d / test {len(test_dates)}d)")
    if best_j is None:
        print("    no qualifying config in-sample")
    else:
        p = grid[best_j]
        tr = trades_full[best_j]
        print(f"    selected: {_fmt(p)}")
        _print_stats("in-sample (train):", trade_stats(tr[tr['date'].isin(train_dates)], len(train_dates)))
        _print_stats("OUT-OF-SAMPLE(test):", trade_stats(tr[tr['date'].isin(test_dates)], len(test_dates)))

    # ---- 2) walk-forward (1y train, 1 quarter test, rolling) ----
    wf = walk_forward(df, grid, inst, train_days=252, test_days=63, metric="sharpe",
                      min_train_trades=MIN_TRAIN_TRADES // 3,
                      trades_full=trades_full, all_dates=all_dates)
    print(f"\n[2] WALK-FORWARD  (252d train / 63d test, rolling -> {len(wf.windows)} windows)")
    _print_stats("aggregate OUT-OF-SAMPLE:", wf.stats)
    if not wf.windows.empty:
        pos = (wf.windows["test_net"] > 0).mean()
        print(f"    windows profitable: {pos:.0%}   "
              f"chosen OR-minutes mix: {dict(wf.windows['or_minutes'].value_counts())}")
        print(f"    trend filter used in {(wf.windows['trend_ma'] > 0).mean():.0%} of windows, "
              f"vol filter in {(wf.windows['vol_min_frac'] > 0).mean():.0%}, "
              f"cutoff 11:30 in {(wf.windows['cutoff'] == '11:30').mean():.0%}")

    if plot:
        _plot(df, inst, grid, best_j, trades_full, wf, split, plot)
    return wf


def _plot(df, inst, grid, best_j, trades_full, wf, split, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    if best_j is not None:
        eq = trades_full[best_j].sort_values("exit_time").set_index("exit_time")["pnl"].cumsum()
        ax.plot(eq.index, eq.values, lw=1.3, color="steelblue",
                label="split-selected config (full history)")
    wfeq = wf.equity_curve()
    if not wfeq.empty:
        ax.plot(wfeq.index, wfeq.values, lw=1.7, color="darkgreen",
                label="walk-forward (all out-of-sample)")
    ax.axvline(pd.Timestamp(split, tz=df.index.tz), color="red", ls="--", lw=1.1, label="split")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"qstack ORB (filtered) — {inst.name} 1-lot cumulative P&L")
    ax.set_ylabel("cumulative net P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\n    saved equity curve -> {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--instrument", required=True, choices=sorted(INSTRUMENTS))
    p.add_argument("--split", default="2025-01-01")
    p.add_argument("--plot", default=None)
    args = p.parse_args(argv)
    run(args.csv, INSTRUMENTS[args.instrument], args.split, args.plot)


if __name__ == "__main__":
    main()
