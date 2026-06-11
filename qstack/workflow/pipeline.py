"""End-to-end pipeline orchestration.

Ties QSConnect -> QSResearch -> Omega together. Each stage is a plain method so
you can run the whole flow or step through it. Sensible zero-config defaults
(synthetic source, local store, paper broker) mean ``Pipeline(...).run()`` works
immediately; pass live components to point it at real data and a real broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Callable

import pandas as pd

from qstack.connect import DataStore, get_source
from qstack.connect.sources import Source
from qstack.omega import Order, PaperBroker
from qstack.omega.broker import Broker, Side
from qstack.research import Backtest, sma_crossover

Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass
class PipelineResult:
    symbol: str
    bars: int
    backtest: "object"          # BacktestResult
    target_position: float      # latest desired position in {-1, 0, +1}
    order: Order | None         # order sent to reach the target (None if no change)

    def __repr__(self) -> str:
        order = "none" if self.order is None else f"{self.order.side.value} {self.order.qty:g}"
        return (f"PipelineResult(symbol={self.symbol!r}, bars={self.bars}, "
                f"target={self.target_position:+.0f}, order={order}, "
                f"sharpe={self.backtest.stats['sharpe']:.2f})")


class Pipeline:
    def __init__(
        self,
        symbol: str,
        strategy: Strategy | None = None,
        source: Source | None = None,
        store: DataStore | None = None,
        broker: Broker | None = None,
        backtest: Backtest | None = None,
        timeframe: str = "1d",
        trade_qty: float = 1.0,
    ):
        self.symbol = symbol
        self.strategy = strategy or partial(sma_crossover, fast=20, slow=50)
        self.source = source or get_source("synthetic")
        self.store = store or DataStore(":memory:")
        self.broker = broker or PaperBroker()
        self.backtest = backtest or Backtest()
        self.timeframe = timeframe
        self.trade_qty = trade_qty

    # -- stages ------------------------------------------------------------
    def ingest(self, start, end) -> int:
        return self.store.ingest(self.source, self.symbol, start, end, self.timeframe)

    def research(self, start=None, end=None):
        df = self.store.read(self.symbol, start, end)
        if df.empty:
            raise ValueError(f"no data for {self.symbol!r}; run ingest() first")
        signal = self.strategy(df)
        return df, signal, self.backtest.run(df, signal)

    def execute(self, df: pd.DataFrame, signal: pd.Series) -> tuple[float, Order | None]:
        target = float(signal.reindex(df.index).ffill().fillna(0.0).iloc[-1])
        price = float(df["close"].iloc[-1])

        current = self.broker.position(self.symbol).qty
        desired = target * self.trade_qty
        delta = desired - current
        if abs(delta) < 1e-12:
            return target, None

        order = Order(
            symbol=self.symbol,
            side=Side.BUY if delta > 0 else Side.SELL,
            qty=abs(delta),
        )
        self.broker.submit(order, price=price)
        return target, order

    # -- full run ----------------------------------------------------------
    def run(self, start=None, end=None, lookback_days: int = 365) -> PipelineResult:
        end = pd.Timestamp(end or datetime.now(timezone.utc))
        start = pd.Timestamp(start or (end - timedelta(days=lookback_days)))

        self.ingest(start, end)
        df, signal, result = self.research(start, end)
        target, order = self.execute(df, signal)

        return PipelineResult(
            symbol=self.symbol,
            bars=len(df),
            backtest=result,
            target_position=target,
            order=order,
        )
