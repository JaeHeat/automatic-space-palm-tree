"""End-to-end demo: ingest -> research -> execute, with no external services.

Run it:  python examples/end_to_end.py
"""

from functools import partial

from qstack import Backtest, DataStore, PaperBroker, Pipeline, get_source
from qstack.research import sma_crossover


def main():
    pipeline = Pipeline(
        symbol="BTC-DEMO",
        strategy=partial(sma_crossover, fast=20, slow=50),
        source=get_source("synthetic"),   # swap for get_source("ccxt") with the crypto extra
        store=DataStore(":memory:"),
        broker=PaperBroker(cash=100_000),
        backtest=Backtest(fee=0.0005, slippage=0.0005),
    )

    result = pipeline.run(lookback_days=730)

    print(result)
    print("\nBacktest stats:")
    print(result.backtest.summary().to_string())
    print(f"\nStore backend in use: {pipeline.store.backend}")
    print(f"Paper broker equity:  {pipeline.broker.equity():,.2f}")


if __name__ == "__main__":
    main()
