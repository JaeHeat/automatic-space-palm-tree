"""Grid-search SMA-crossover windows and rank configurations by Sharpe.

    python examples/param_sweep.py --symbol SPY --source yfinance --start 2015-01-01
    python examples/param_sweep.py --symbol BTC-USD --source yfinance     # crypto via Yahoo

Prints the top configurations and writes a Sharpe heatmap (fast vs slow window).
"""

from __future__ import annotations

import argparse

import pandas as pd

from qstack import Backtest, DataStore, get_source
from qstack.research import grid_search, sma_crossover


def run(symbol, source_name, start, end, fee, slippage):
    store = DataStore(":memory:")
    n = store.ingest(get_source(source_name), symbol, start, end)
    df = store.read(symbol, start, end)
    print(f"Loaded {n} bars of {symbol} "
          f"({df.index.min().date()} -> {df.index.max().date()})\n")

    grid = {
        "fast": [5, 10, 15, 20, 30, 40, 50],
        "slow": [50, 75, 100, 150, 200, 250],
    }
    table = grid_search(
        df, sma_crossover, grid,
        backtest=Backtest(fee=fee, slippage=slippage),
        rank_by="sharpe",
        valid=lambda p: p["fast"] < p["slow"],   # fast must be shorter than slow
    )
    return df, table


def heatmap(table: pd.DataFrame, symbol: str, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("(matplotlib not installed -- skipping heatmap)")
        return None

    pivot = table.pivot(index="fast", columns="slow", values="sharpe")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("slow window")
    ax.set_ylabel("fast window")
    ax.set_title(f"qstack SMA-crossover Sharpe heatmap — {symbol}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved Sharpe heatmap -> {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--source", default="yfinance", help="yfinance | synthetic | ccxt")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--plot", default="param_sweep_heatmap.png")
    args = p.parse_args(argv)

    df, table = run(args.symbol, args.source, args.start, args.end, args.fee, args.slippage)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print(f"Top {args.top} configurations by Sharpe:\n")
    print(table.head(args.top).to_string(index=False))
    print(f"\n... of {len(table)} valid combinations tested.")

    if args.plot:
        heatmap(table, args.symbol, args.plot)


if __name__ == "__main__":
    main()
