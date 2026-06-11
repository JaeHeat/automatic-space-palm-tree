"""QSResearch equivalent — research and run trading strategies.

    strategies   signal generators that map an OHLCV frame -> target position
                 series in {-1, 0, +1}. Ships with sma_crossover, rsi_reversion,
                 and momentum.
    Backtest     a pure-numpy vectorized backtester with fees + slippage and a
                 standard performance report (CAGR, Sharpe, max drawdown, ...).
                 Uses vectorbt automatically if it is installed.
"""

from qstack.research.backtest import Backtest, BacktestResult
from qstack.research.strategies import momentum, rsi_reversion, sma_crossover

__all__ = [
    "Backtest",
    "BacktestResult",
    "sma_crossover",
    "rsi_reversion",
    "momentum",
]
