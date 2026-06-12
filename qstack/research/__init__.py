"""QSResearch equivalent — research and run trading strategies.

    strategies   signal generators that map an OHLCV frame -> target position
                 series in {-1, 0, +1}. Ships with sma_crossover, rsi_reversion,
                 and momentum.
    Backtest     a pure-numpy vectorized backtester with fees + slippage and a
                 standard performance report (CAGR, Sharpe, max drawdown, ...).
                 Uses vectorbt automatically if it is installed.
"""

from qstack.research.backtest import Backtest, BacktestResult
from qstack.research.optimize import grid_search
from qstack.research.orb import (
    ES,
    NQ,
    Instrument,
    ORBParams,
    ORBResult,
    WalkForwardResult,
    backtest_orb,
    trade_stats,
    walk_forward,
)
from qstack.research.strategies import momentum, rsi_reversion, sma_crossover

__all__ = [
    "Backtest",
    "BacktestResult",
    "grid_search",
    "sma_crossover",
    "rsi_reversion",
    "momentum",
    "backtest_orb",
    "walk_forward",
    "trade_stats",
    "ORBParams",
    "ORBResult",
    "WalkForwardResult",
    "Instrument",
    "NQ",
    "ES",
]
