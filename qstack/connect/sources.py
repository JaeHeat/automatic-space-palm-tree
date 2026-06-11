"""Market-data sources.

Every source exposes ``fetch(symbol, start, end, timeframe) -> DataFrame`` with a
normalised schema: a ``timestamp`` index (UTC) and ``open/high/low/close/volume``
columns. That uniform shape is what lets the store, backtester, and broker stay
backend-agnostic.

The default ``SyntheticSource`` needs no network and no API key, so the whole
stack is runnable out of the box. ``get_source("yfinance")`` and
``get_source("ccxt")`` pull live data when those optional deps are installed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class Source:
    """Base class. Subclasses implement :meth:`fetch`."""

    name = "base"

    def fetch(self, symbol, start, end, timeframe="1d") -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=str.lower)
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"source returned data missing columns: {missing}")
        df = df[OHLCV_COLUMNS].copy()
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC") \
            if df.index.tz is not None else pd.DatetimeIndex(df.index).tz_localize("UTC")
        df.index.name = "timestamp"
        return df.sort_index()


class SyntheticSource(Source):
    """Deterministic geometric-brownian-motion OHLCV generator.

    Seeded off the symbol name so a given symbol always yields the same series --
    handy for reproducible tests and demos without hitting any API.
    """

    name = "synthetic"

    def __init__(self, start_price=100.0, annual_vol=0.35, annual_drift=0.08):
        self.start_price = start_price
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift

    def fetch(self, symbol, start, end, timeframe="1d") -> pd.DataFrame:
        start, end = _utc(start), _utc(end)
        freq = {"1d": "D", "1h": "H", "1m": "min"}.get(timeframe, "D")
        index = pd.date_range(start, end, freq=freq, tz="UTC")
        if len(index) == 0:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        periods_per_year = {"D": 252, "H": 252 * 24, "min": 252 * 24 * 60}[freq]
        dt = 1.0 / periods_per_year
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))

        shocks = rng.normal(
            (self.annual_drift - 0.5 * self.annual_vol**2) * dt,
            self.annual_vol * np.sqrt(dt),
            size=len(index),
        )
        close = self.start_price * np.exp(np.cumsum(shocks))
        open_ = np.concatenate([[self.start_price], close[:-1]])
        intrabar = np.abs(rng.normal(0, self.annual_vol * np.sqrt(dt), len(index)))
        high = np.maximum(open_, close) * (1 + intrabar)
        low = np.minimum(open_, close) * (1 - intrabar)
        volume = rng.integers(1_000, 1_000_000, len(index)).astype(float)

        return self._normalise(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=index,
            )
        )


class YFinanceSource(Source):
    """Equities/ETF data via the optional ``yfinance`` package."""

    name = "yfinance"

    def fetch(self, symbol, start, end, timeframe="1d") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "yfinance not installed. Run: pip install 'qstack[data]'"
            ) from exc

        interval = {"1d": "1d", "1h": "60m", "1m": "1m"}.get(timeframe, "1d")
        df = yf.download(
            symbol, start=_utc(start), end=_utc(end), interval=interval,
            auto_adjust=True, progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return self._normalise(df)


class CCXTSource(Source):
    """Crypto data via the optional ``ccxt`` package (defaults to Binance)."""

    name = "ccxt"

    def __init__(self, exchange="binance"):
        self.exchange = exchange

    def fetch(self, symbol, start, end, timeframe="1d") -> pd.DataFrame:
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "ccxt not installed. Run: pip install 'qstack[crypto]'"
            ) from exc

        client = getattr(ccxt, self.exchange)()
        since = int(_utc(start).timestamp() * 1000)
        end_ms = int(_utc(end).timestamp() * 1000)
        rows = []
        while since < end_ms:
            batch = client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + 1
            if len(batch) < 1000:
                break

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return self._normalise(df.set_index("timestamp"))


_SOURCES = {
    "synthetic": SyntheticSource,
    "yfinance": YFinanceSource,
    "ccxt": CCXTSource,
}


def get_source(name: str = "synthetic", **kwargs) -> Source:
    """Return a configured data source by name.

    >>> get_source("synthetic").fetch("BTC", "2024-01-01", "2024-03-01").shape[1]
    5
    """
    try:
        return _SOURCES[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown source {name!r}; choose from {sorted(_SOURCES)}"
        ) from None
