"""qstack — an open-source quant stack.

Open-source equivalents of the four commercial Quant Science libraries:

    QSConnect  -> qstack.connect   (research data store + market data sources)
    QSResearch -> qstack.research  (strategies + vectorized backtesting)
    QSWorkflow -> qstack.workflow  (end-to-end pipeline orchestration)
    Omega      -> qstack.omega     (order execution: paper + live brokers)

The package runs with zero optional dependencies (synthetic data, SQLite store,
pure-numpy backtester). Install extras to swap in live backends -- see README.
"""

from qstack.connect import DataStore, get_source
from qstack.omega import PaperBroker, Order
from qstack.research import Backtest, sma_crossover
from qstack.workflow import Pipeline

__version__ = "0.1.0"

__all__ = [
    "DataStore",
    "get_source",
    "Backtest",
    "sma_crossover",
    "PaperBroker",
    "Order",
    "Pipeline",
    "__version__",
]
