"""Keyless FRED data access.

FRED exposes a CSV download endpoint that requires no API key:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

This module wraps it with on-disk caching and retry/backoff so the rest of the
project has a single, reliable data source for every series we need (BTC price,
Nasdaq-100, M2 components, FX rates, business-cycle indicators).
"""

from __future__ import annotations

import io
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Cache lives next to the repo's data/ dir by default.
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

_HEADERS = {
    # FRED occasionally rate-limits clients with no UA; a plain browser-ish UA
    # is friendlier and avoids the 000/empty responses seen with bare curl.
    "User-Agent": "cryptomacro/0.1 (+https://github.com; research use)"
}


def _cache_path(series_id: str) -> Path:
    return _CACHE_DIR / f"{series_id}.csv"


def fetch_series(
    series_id: str,
    *,
    use_cache: bool = True,
    max_age_hours: float = 12.0,
    retries: int = 4,
    timeout: float = 20.0,
) -> pd.Series:
    """Return a single FRED series as a float Series indexed by date.

    Parameters
    ----------
    series_id:
        FRED series identifier, e.g. ``"CBBTCUSD"`` or ``"M2SL"``.
    use_cache:
        If True, reuse a cached CSV younger than ``max_age_hours`` before
        hitting the network. Cached data also serves as a fallback if every
        network attempt fails.
    max_age_hours:
        Maximum age of a cached file before it is considered stale.
    retries:
        Number of network attempts; backoff is 2s, 4s, 8s, 16s.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(series_id)

    fresh_cache = (
        use_cache
        and path.exists()
        and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
    )
    if fresh_cache:
        return _parse_csv(path.read_text(), series_id)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            text = _download(series_id, timeout=timeout)
            if not text.strip() or "observation_date" not in text.split("\n", 1)[0]:
                raise ValueError(f"Unexpected FRED payload for {series_id!r}")
            path.write_text(text)
            return _parse_csv(text, series_id)
        except Exception as err:  # noqa: BLE001 - retry on any network/parse error
            last_err = err
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))

    # Network exhausted: fall back to any cached copy, even if stale.
    if path.exists():
        return _parse_csv(path.read_text(), series_id)
    raise RuntimeError(f"Could not fetch FRED series {series_id!r}: {last_err}")


def _download(series_id: str, *, timeout: float) -> str:
    """Fetch a series' raw CSV text.

    Tries the ``requests`` library first (the natural path for local users). Some
    sandboxed/proxied environments hang ``requests`` on read while ``curl`` works
    fine, so we fall back to a ``curl`` subprocess if requests fails.
    """
    try:
        resp = requests.get(
            FRED_CSV_URL,
            params={"id": series_id},
            headers=_HEADERS,
            timeout=(min(timeout, 10), timeout),
        )
        resp.raise_for_status()
        if "observation_date" in resp.text.split("\n", 1)[0]:
            return resp.text
    except Exception:  # noqa: BLE001 - fall through to curl
        pass

    proc = subprocess.run(
        ["curl", "-sS", "-m", str(int(timeout)),
         "-A", _HEADERS["User-Agent"],
         f"{FRED_CSV_URL}?id={series_id}"],
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _parse_csv(text: str, series_id: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]  # "observation_date" (or legacy "DATE")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    # FRED uses "." for missing observations -> coerce to NaN floats.
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.name = series_id
    s.index.name = "date"
    return s.dropna()


def fetch_series_chunked(
    series_id: str,
    *,
    start_year: int = 2014,
    end_year: int | None = None,
    use_cache: bool = True,
    max_age_hours: float = 12.0,
    timeout: float = 15.0,
) -> pd.Series:
    """Fetch a (large, daily) series in one-year chunks and assemble it.

    On slow or rate-limited links a full daily-series download can exceed the
    timeout, while small per-year requests succeed. This fetches
    ``[start_year, end_year]`` a year at a time (with retries per chunk), writes
    the assembled CSV to the normal cache, and returns the combined Series. Use
    it for daily series like ``NASDAQ100`` or ``T10Y2Y`` when ``fetch_series``
    times out.
    """
    import datetime as _dt

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(series_id)
    if use_cache and path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return _parse_csv(path.read_text(), series_id)

    end_year = end_year or _dt.date.today().year
    pieces: list[pd.Series] = []
    for year in range(start_year, end_year + 1):
        for attempt in range(5):
            try:
                proc = subprocess.run(
                    ["curl", "-sS", "-m", str(int(timeout)), "-A", _HEADERS["User-Agent"],
                     f"{FRED_CSV_URL}?id={series_id}&cosd={year}-01-01&coed={year}-12-31"],
                    capture_output=True, text=True,
                )
                if "observation_date" in proc.stdout.split("\n", 1)[0]:
                    pieces.append(_parse_csv(proc.stdout, series_id))
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

    if not pieces:
        if path.exists():
            return _parse_csv(path.read_text(), series_id)
        raise RuntimeError(f"Could not chunk-fetch FRED series {series_id!r}")

    combined = pd.concat(pieces)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    # Persist in the same CSV shape fetch_series expects.
    out = combined.rename(series_id)
    out.index.name = "observation_date"
    path.write_text(out.to_csv())
    combined.index.name = "date"
    return combined


def fetch_many(series_ids: list[str], **kwargs) -> pd.DataFrame:
    """Fetch several series and return them aligned in one DataFrame (outer join)."""
    cols = {sid: fetch_series(sid, **kwargs) for sid in series_ids}
    return pd.DataFrame(cols)
