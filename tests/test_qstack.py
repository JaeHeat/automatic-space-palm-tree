"""Smoke + behaviour tests for the qstack stack. Run with: pytest -q"""

import pandas as pd
import pytest

from qstack import Backtest, DataStore, Order, PaperBroker, Pipeline, get_source
from qstack.omega.broker import Side
from qstack.research import momentum, rsi_reversion, sma_crossover

START, END = "2022-01-01", "2024-01-01"


# -- connect ---------------------------------------------------------------
def test_synthetic_source_is_deterministic_and_normalised():
    src = get_source("synthetic")
    a = src.fetch("BTC", START, END)
    b = src.fetch("BTC", START, END)
    assert list(a.columns) == ["open", "high", "low", "close", "volume"]
    assert str(a.index.tz) == "UTC"
    pd.testing.assert_frame_equal(a, b)            # same symbol -> same series
    assert not a.equals(src.fetch("ETH", START, END))


def test_store_roundtrip_and_upsert():
    store = DataStore(":memory:")
    src = get_source("synthetic")
    n = store.ingest(src, "BTC", START, END)
    assert n > 0
    assert store.symbols() == ["BTC"]

    back = store.read("BTC")
    assert len(back) == n
    # re-ingesting the same range must not duplicate rows
    store.ingest(src, "BTC", START, END)
    assert len(store.read("BTC")) == n


def test_store_read_window():
    store = DataStore(":memory:")
    store.ingest(get_source("synthetic"), "BTC", START, END)
    windowed = store.read("BTC", "2023-01-01", "2023-06-01")
    assert windowed.index.min() >= pd.Timestamp("2023-01-01", tz="UTC")
    assert windowed.index.max() <= pd.Timestamp("2023-06-01", tz="UTC")


# -- research --------------------------------------------------------------
@pytest.mark.parametrize("strategy", [sma_crossover, rsi_reversion, momentum])
def test_strategies_return_bounded_positions(strategy):
    df = get_source("synthetic").fetch("BTC", START, END)
    sig = strategy(df)
    assert sig.index.equals(df.index)
    assert sig.between(-1, 1).all()
    assert not sig.isna().any()


def test_backtest_has_no_lookahead_and_reports_stats():
    df = get_source("synthetic").fetch("BTC", START, END)
    result = Backtest().run(df, sma_crossover(df))
    # first applied position is always flat because of the one-bar lag
    assert result.positions.iloc[0] == 0.0
    assert set(result.stats) == {
        "total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "n_trades"
    }
    assert result.equity.iloc[0] == pytest.approx(1.0, abs=0.05)


def test_costs_reduce_returns():
    df = get_source("synthetic").fetch("BTC", START, END)
    sig = sma_crossover(df)
    free = Backtest(fee=0, slippage=0).run(df, sig).stats["total_return"]
    costly = Backtest(fee=0.01, slippage=0.01).run(df, sig).stats["total_return"]
    assert costly < free


# -- omega -----------------------------------------------------------------
def test_paper_broker_tracks_position_and_cash():
    broker = PaperBroker(cash=10_000, commission=0, slippage=0)
    broker.submit(Order("BTC", Side.BUY, 2), price=100)
    pos = broker.position("BTC")
    assert pos.qty == 2 and pos.avg_price == 100
    assert broker.cash == pytest.approx(10_000 - 200)

    broker.submit(Order("BTC", Side.SELL, 2), price=110)
    assert broker.position("BTC").is_flat
    assert broker.cash == pytest.approx(10_000 + 20)   # +10 per unit profit


def test_paper_broker_requires_a_price():
    with pytest.raises(ValueError):
        PaperBroker().submit(Order("BTC", Side.BUY, 1))


# -- workflow --------------------------------------------------------------
def test_pipeline_end_to_end():
    pipeline = Pipeline(symbol="BTC", store=DataStore(":memory:"))
    result = pipeline.run(start=START, end=END)
    assert result.bars > 0
    assert result.target_position in (-1.0, 0.0, 1.0)
    if result.order is not None:
        assert pipeline.broker.position("BTC").qty != 0


def test_pipeline_research_requires_data():
    pipeline = Pipeline(symbol="BTC", store=DataStore(":memory:"))
    with pytest.raises(ValueError):
        pipeline.research()
