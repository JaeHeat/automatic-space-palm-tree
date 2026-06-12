"""Loaders for on-disk market data exports.

``load_ohlcv_csv`` reads a Databento-style 5m CSV export (epoch-second ``time``
column plus OHLCV) and returns a clean, timezone-aware OHLCV frame indexed in
US/Eastern — the shape the ORB engine and the rest of qstack expect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ohlcv_csv(
    path: str | Path,
    tz: str = "America/New_York",
    time_col: str = "time",
    time_unit: str = "s",
) -> pd.DataFrame:
    """Load an epoch-timestamped OHLCV CSV into a tz-aware frame.

    Args:
        path:      CSV file with a ``time`` column (epoch) + open/high/low/close
                   and an optional volume column.
        tz:        target timezone for the returned index.
        time_col:  name of the epoch timestamp column.
        time_unit: unit of the epoch column ("s", "ms", "ns").

    Returns:
        DataFrame indexed by a tz-aware DatetimeIndex (named "timestamp") with
        lower-cased open/high/low/close[/volume] columns, sorted and de-duped.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    idx = pd.to_datetime(df[time_col], unit=time_unit, utc=True).dt.tz_convert(tz)
    df = df.drop(columns=[time_col])

    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    if "volume" not in df.columns:
        for c in list(df.columns):                 # tolerate "Volume", "vol", etc.
            if c.lower().startswith("volume") and c != "volume":
                df = df.rename(columns={c: "volume"})
                break

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df[keep].copy()
    out.index = pd.DatetimeIndex(idx.values, tz="UTC").tz_convert(tz)
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=[c for c in ("open", "high", "low", "close") if c in out.columns])
