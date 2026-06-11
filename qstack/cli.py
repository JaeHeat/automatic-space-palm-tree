"""Command-line entry point: run an end-to-end pipeline from the terminal.

    qstack run --symbol BTC --strategy sma --fast 20 --slow 50
    qstack run --symbol AAPL --source yfinance --lookback 730
"""

from __future__ import annotations

import argparse
from functools import partial

from qstack.connect import DataStore, get_source
from qstack.research import momentum, rsi_reversion, sma_crossover
from qstack.workflow import Pipeline

_STRATEGIES = {"sma": sma_crossover, "rsi": rsi_reversion, "momentum": momentum}


def _build_strategy(args):
    if args.strategy == "sma":
        return partial(sma_crossover, fast=args.fast, slow=args.slow)
    return _STRATEGIES[args.strategy]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="qstack", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an end-to-end pipeline")
    run.add_argument("--symbol", required=True)
    run.add_argument("--source", default="synthetic", help="synthetic | yfinance | ccxt")
    run.add_argument("--strategy", default="sma", choices=sorted(_STRATEGIES))
    run.add_argument("--fast", type=int, default=20)
    run.add_argument("--slow", type=int, default=50)
    run.add_argument("--timeframe", default="1d")
    run.add_argument("--lookback", type=int, default=365, help="lookback in days")
    run.add_argument("--db", default=":memory:", help="store path (default in-memory)")

    args = parser.parse_args(argv)

    pipeline = Pipeline(
        symbol=args.symbol,
        strategy=_build_strategy(args),
        source=get_source(args.source),
        store=DataStore(args.db),
        timeframe=args.timeframe,
    )
    result = pipeline.run(lookback_days=args.lookback)

    print(f"\nqstack run — {result.symbol} ({args.strategy}, {result.bars} bars)\n")
    for k, v in result.backtest.stats.items():
        print(f"  {k:<16} {v:>10.4f}")
    order = "no change" if result.order is None else \
        f"{result.order.side.value} {result.order.qty:g}"
    print(f"\n  target position  {result.target_position:>+10.0f}")
    print(f"  order sent       {order:>10}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
