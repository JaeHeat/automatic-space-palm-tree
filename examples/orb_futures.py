"""Find a profitable ORB day-trading strategy on ES/NQ futures, honestly.

Optimizes the opening-range-breakout parameters on an in-sample TRAIN window,
then reports performance on a held-out out-of-sample TEST window so the result
isn't just curve-fit. Sized in real 1-lot futures economics (point value,
commission, slippage).

    python examples/orb_futures.py --csv nq.csv --instrument NQ --split 2025-01-01

CSV format: epoch-second `time` column plus open/high/low/close[/volume]
(Databento 5m export).
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import ES, NQ, ORBParams, backtest_orb
from qstack.research.orb import Instrument, day_groups

INSTRUMENTS = {"NQ": NQ, "ES": ES}

# Search grid. Kept deliberately small/coarse to limit the number of in-sample
# bets we make (every extra knob is another chance to overfit 4 years of data).
GRID = {
    "or_minutes": [15, 30, 60],
    "direction": ["both", "long", "short"],
    "stop_mult": [0.5, 1.0, 1.5],
    "target_mult": [1.0, 2.0, 3.0, None],
}

MIN_TRAIN_TRADES = 150   # ignore configs that barely trade in-sample


def optimize(df_train: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    rows = []
    keys = list(GRID)
    groups = day_groups(df_train)   # compute the date grouping once, reuse per config
    for combo in itertools.product(*(GRID[k] for k in keys)):
        p = dict(zip(keys, combo))
        res = backtest_orb(df_train, ORBParams(**p), inst, groups=groups)
        s = res.stats
        if s["n_trades"] >= MIN_TRAIN_TRADES:
            rows.append({**p, **s})
    table = pd.DataFrame(rows)
    # Rank by risk-adjusted return (Sharpe), require a positive edge.
    table = table[table["profit_factor"] > 1.0]
    return table.sort_values("sharpe", ascending=False).reset_index(drop=True)


def run(csv: str, inst: Instrument, split: str):
    df = load_ohlcv_csv(csv)
    split_ts = pd.Timestamp(split, tz=df.index.tz)
    train, test = df[df.index < split_ts], df[df.index >= split_ts]
    print(f"\n=== {inst.name} ===")
    print(f"loaded {len(df):,} bars {df.index.min().date()} -> {df.index.max().date()}")
    print(f"train: {train.index.min().date()} -> {train.index.max().date()}  ({len(train):,} bars)")
    print(f"test : {test.index.min().date()} -> {test.index.max().date()}  ({len(test):,} bars)\n")

    ranked = optimize(train, inst)
    if ranked.empty:
        print("No configuration cleared the in-sample filters.")
        return None
    pd.set_option("display.width", 220)
    show = ["or_minutes", "direction", "stop_mult", "target_mult",
            "n_trades", "net_pnl", "profit_factor", "win_rate", "max_drawdown", "sharpe"]
    print("Top 5 IN-SAMPLE (train) configurations by Sharpe:\n")
    print(ranked[show].head(5).to_string(index=False))

    best = ranked.iloc[0]
    params = ORBParams(or_minutes=int(best["or_minutes"]), direction=best["direction"],
                       stop_mult=float(best["stop_mult"]),
                       target_mult=None if pd.isna(best["target_mult"]) else float(best["target_mult"]))

    test_res = backtest_orb(test, params, inst)
    print("\n>>> Selected config (best train Sharpe):")
    print(f"    OR={params.or_minutes}m  dir={params.direction}  "
          f"stop={params.stop_mult}xRange  target={params.target_mult}xRange")
    print("\n    IN-SAMPLE (train):")
    _print_stats(backtest_orb(train, params, inst).stats)
    print("\n    OUT-OF-SAMPLE (test):")
    _print_stats(test_res.stats)
    return params, test_res, df


def _print_stats(s: dict):
    print(f"      trades={s['n_trades']}  net=${s['net_pnl']:,.0f}  "
          f"PF={s['profit_factor']:.2f}  win={s['win_rate']:.1%}  "
          f"avg=${s['avg_trade']:,.0f}/trade  maxDD=${s['max_drawdown']:,.0f}  "
          f"sharpe={s['sharpe']:.2f}")


def plot(df, params, inst, split, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping plot)")
        return None
    full = backtest_orb(df, params, inst)
    eq = full.equity_curve()
    split_ts = pd.Timestamp(split, tz=df.index.tz)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(eq.index, eq.values, lw=1.4)
    ax.axvline(split_ts, color="red", ls="--", lw=1.2, label="train/test split")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"qstack ORB — {inst.name} 1-lot cumulative P&L "
                 f"(OR{params.or_minutes}m {params.direction} stop{params.stop_mult} tgt{params.target_mult})")
    ax.set_ylabel("cumulative net P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved equity curve -> {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--instrument", required=True, choices=sorted(INSTRUMENTS))
    p.add_argument("--split", default="2025-01-01", help="train/test boundary date")
    p.add_argument("--plot", default=None)
    args = p.parse_args(argv)

    inst = INSTRUMENTS[args.instrument]
    out = run(args.csv, inst, args.split)
    if out and args.plot:
        params, _, df = out
        plot(df, params, inst, args.split, args.plot)


if __name__ == "__main__":
    main()
