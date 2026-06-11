"""QSConnect equivalent — build and query your quant research database.

Two pieces:

    DataStore   a local OHLCV store. Uses DuckDB when installed, otherwise a
                zero-dependency SQLite backend. Same API either way.
    sources     pluggable market-data fetchers (synthetic / yfinance / ccxt)
                that all return a normalised OHLCV DataFrame.
"""

from qstack.connect.sources import SyntheticSource, get_source
from qstack.connect.store import DataStore

__all__ = ["DataStore", "get_source", "SyntheticSource"]
