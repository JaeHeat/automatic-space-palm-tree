"""Local OHLCV research database.

Prefers DuckDB (fast columnar, great for analytics) when it is installed, and
transparently falls back to the stdlib ``sqlite3`` module otherwise so the store
always works. Both backends present the same API:

    store.write(symbol, df)         upsert a normalised OHLCV frame
    store.read(symbol, start, end)  read back a frame
    store.symbols()                 list stored symbols
    store.ingest(source, symbol, ...) fetch from a source and write in one call
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from qstack.connect.sources import OHLCV_COLUMNS, Source

_TABLE = "ohlcv"


def _as_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class DataStore:
    def __init__(self, path: str | Path = "qstack.db"):
        self.path = str(path)
        self._duckdb = self._open_duckdb()
        self._sqlite = None
        if self._duckdb is None:
            self._init_sqlite()

    # -- backend selection -------------------------------------------------
    def _open_duckdb(self):
        try:
            import duckdb
        except ImportError:
            return None
        con = duckdb.connect(self.path if self.path != ":memory:" else ":memory:")
        con.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    symbol VARCHAR, timestamp TIMESTAMP,
                    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
                    PRIMARY KEY (symbol, timestamp))"""
        )
        return con

    def _init_sqlite(self):
        # Keep one connection for the lifetime of the store: an in-memory
        # database only exists for as long as its connection is open.
        self._sqlite = sqlite3.connect(self.path)
        self._sqlite.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    symbol TEXT, timestamp TEXT,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (symbol, timestamp))"""
        )
        self._sqlite.commit()

    @property
    def backend(self) -> str:
        return "duckdb" if self._duckdb is not None else "sqlite"

    # -- writes ------------------------------------------------------------
    def write(self, symbol: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        frame = df[OHLCV_COLUMNS].copy()
        frame.insert(0, "timestamp", pd.DatetimeIndex(df.index).tz_convert("UTC"))
        frame.insert(0, "symbol", symbol)

        if self._duckdb is not None:
            self._duckdb.execute(
                f"DELETE FROM {_TABLE} WHERE symbol = ? AND timestamp BETWEEN ? AND ?",
                [symbol, frame["timestamp"].min(), frame["timestamp"].max()],
            )
            self._duckdb.register("_incoming", frame)
            self._duckdb.execute(f"INSERT INTO {_TABLE} SELECT * FROM _incoming")
            self._duckdb.unregister("_incoming")
        else:
            frame = frame.copy()
            frame["timestamp"] = frame["timestamp"].astype(str)
            self._sqlite.executemany(
                f"INSERT OR REPLACE INTO {_TABLE} VALUES (?,?,?,?,?,?,?)",
                frame.itertuples(index=False, name=None),
            )
            self._sqlite.commit()
        return len(frame)

    def ingest(self, source: Source, symbol: str, start, end, timeframe="1d") -> int:
        """Fetch from a data source and persist in one step."""
        return self.write(symbol, source.fetch(symbol, start, end, timeframe))

    # -- reads -------------------------------------------------------------
    def read(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        clauses, params = ["symbol = ?"], [symbol]
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(str(_as_utc(start)))
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(str(_as_utc(end)))
        where = " AND ".join(clauses)
        sql = f"SELECT timestamp, {', '.join(OHLCV_COLUMNS)} FROM {_TABLE} WHERE {where} ORDER BY timestamp"

        if self._duckdb is not None:
            df = self._duckdb.execute(sql, params).fetch_df()
        else:
            df = pd.read_sql_query(sql, self._sqlite, params=params)

        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    def symbols(self) -> list[str]:
        sql = f"SELECT DISTINCT symbol FROM {_TABLE} ORDER BY symbol"
        if self._duckdb is not None:
            return [r[0] for r in self._duckdb.execute(sql).fetchall()]
        return [r[0] for r in self._sqlite.execute(sql).fetchall()]

    def close(self):
        if self._duckdb is not None:
            self._duckdb.close()
        if self._sqlite is not None:
            self._sqlite.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
