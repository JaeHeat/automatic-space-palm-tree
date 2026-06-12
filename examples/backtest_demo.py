"""Run a real backtest with qstack and compare strategies to buy & hold.

    python examples/backtest_demo.py --symbol SPY --source yfinance --start 2018-01-01
    python examples/backtest_demo.py --symbol BTC --source synthetic   # offline

Prints a side-by-side performance table and writes an equity-curve PNG.
"""

from __future__ import annotations

import argparse
from functools import partial

import pandas as pd

from qstack import Backtest, DataStore, get_source
from qstack.research import momentum, rsi_reversion, sma_crossover

# strategy name -> signal callable
STRATEGIES = {
    "sma_20_50": partial(sma_crossover, fast=20, slow=50),
    "sma_50_200": partial(sma_crossover, fast=50, slow=200),
    "rsi_reversion": rsi_reversion,
    "momentum_90": partial(momentum, lookback=90),
}


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Benchmark: always fully long."""
    return pd.Series(1.0, index=df.index, name="position")


def run(symbol: str, source_name: str, start: str, end: str,
        fee: float, slippage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1) QSConnect: ingest into the research store, then read it back
    store = DataStore(":memory:")
    n = store.ingest(get_source(source_name), symbol, start, end)
    df = store.read(symbol, start, end)
    print(f"Loaded {n} bars of {symbol} from '{source_name}' "
          f"({df.index.min().date()} -> {df.index.max().date()}), "
          f"store backend = {store.backend}\n")

    # 2) QSResearch: backtest every strategy + the benchmark on the same data
    engine = Backtest(fee=fee, slippage=slippage)
    rows, equity_curves = [], {}
    for name, strat in {"buy_and_hold": buy_and_hold, **STRATEGIES}.items():
        res = engine.run(df, strat(df))
        equity_curves[name] = res.equity
        rows.append({"strategy": name, **res.stats})

    table = pd.DataFrame(rows).set_index("strategy")
    return table, pd.DataFrame(equity_curves)


def plot(equity: pd.DataFrame, symbol: str, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping plot; pip install matplotlib)")
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in equity.columns:
        style = dict(lw=2.4, color="black", ls="--") if name == "buy_and_hold" else dict(lw=1.6)
        ax.plot(equity.index, equity[name], label=name, **style)
    ax.set_title(f"qstack backtest — {symbol} (growth of $1, net of costs)")
    ax.set_ylabel("equity (x starting capital)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved equity-curve chart -> {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--source", default="yfinance", help="yfinance | synthetic | ccxt")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--plot", default="backtest_equity.png")
    args = p.parse_args(argv)

    table, equity = run(args.symbol, args.source, args.start, args.end,
                        args.fee, args.slippage)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("Performance (sorted by Sharpe):\n")
    print(table.sort_values("sharpe", ascending=False).to_string())

    if args.plot:
        plot(equity, args.symbol, args.plot)


if __name__ == "__main__":
    main()
