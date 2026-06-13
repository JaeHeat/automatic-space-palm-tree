"""Reusable time-series transforms shared by all four analyses.

These are deliberately small, pure functions on pandas objects so they can be
unit-reasoned and reused across scripts and notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def returns(price: pd.Series, *, log: bool = False) -> pd.Series:
    """Simple (default) or log returns of a price series."""
    if log:
        return np.log(price / price.shift(1)).dropna()
    return price.pct_change().dropna()


def resample_returns(price: pd.Series, freq: str = "W", *, log: bool = False) -> pd.Series:
    """Period returns at ``freq`` (e.g. 'W' weekly, 'ME' monthly) from a price series."""
    p = price.resample(freq).last()
    return returns(p, log=log)


# ---------------------------------------------------------------------------
# Drawdowns & cycle shape
# ---------------------------------------------------------------------------

def drawdown(price: pd.Series) -> pd.Series:
    """Percent drawdown from the running all-time high (0 at a new high, negative below)."""
    running_max = price.cummax()
    return (price / running_max - 1.0) * 100.0


def rebase(price: pd.Series, base: float = 100.0) -> pd.Series:
    """Rebase a price series so its first valid point equals ``base``."""
    price = price.dropna()
    return price / price.iloc[0] * base


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 90) -> pd.Series:
    """Rolling Pearson correlation of two aligned series over ``window`` observations."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).dropna()


def lead_lag_correlation(
    driver: pd.Series,
    target: pd.Series,
    max_lag: int = 26,
) -> pd.DataFrame:
    """Cross-correlation of ``target`` against ``driver`` shifted forward by 0..max_lag.

    A positive lag means *driver leads target* (driver moves first). The lag with
    the highest correlation is the estimated lead time, in whatever frequency the
    inputs share (e.g. weeks if you pass weekly series).

    Returns a DataFrame with columns ``lag`` and ``corr``.
    """
    df = pd.concat([driver, target], axis=1, join="inner").dropna()
    d, t = df.iloc[:, 0], df.iloc[:, 1]
    rows = []
    for lag in range(0, max_lag + 1):
        rows.append({"lag": lag, "corr": d.shift(lag).corr(t)})
    return pd.DataFrame(rows)


def best_lead_lag(driver: pd.Series, target: pd.Series, max_lag: int = 26) -> tuple[int, float]:
    """Return ``(lag, corr)`` maximising correlation (driver leading target)."""
    cc = lead_lag_correlation(driver, target, max_lag=max_lag)
    row = cc.loc[cc["corr"].idxmax()]
    return int(row["lag"]), float(row["corr"])


# ---------------------------------------------------------------------------
# Regimes (bull / bear)
# ---------------------------------------------------------------------------

def regime_by_ma(price: pd.Series, window: int = 200) -> pd.Series:
    """Label each day 'bull' (price >= rolling MA) or 'bear' (below). Daily series in."""
    ma = price.rolling(window).mean()
    label = np.where(price >= ma, "bull", "bear")
    return pd.Series(label, index=price.index, name="regime").reindex(price.index).where(ma.notna())


def regime_by_drawdown(price: pd.Series, bear_threshold: float = -20.0) -> pd.Series:
    """Label 'bear' once drawdown from ATH breaches ``bear_threshold`` (%), else 'bull'."""
    dd = drawdown(price)
    return pd.Series(np.where(dd <= bear_threshold, "bear", "bull"), index=price.index, name="regime")


# ---------------------------------------------------------------------------
# Cycle alignment
# ---------------------------------------------------------------------------

def align_to_anchor(
    price: pd.Series,
    anchors: dict[str, pd.Timestamp],
    *,
    window_days: int = 1100,
    pre_days: int = 0,
    rebase_at_anchor: bool = True,
) -> pd.DataFrame:
    """Align multiple cycles on a common x-axis of 'days since anchor'.

    For each label/anchor, slice ``[anchor - pre_days, anchor + window_days]`` and
    index it by integer day offset. Optionally rebase each slice to 100 at the
    anchor so cycles overlay regardless of absolute price.

    Returns a DataFrame indexed by ``day_offset`` with one column per cycle label.
    """
    out = {}
    for label, anchor in anchors.items():
        lo = anchor - pd.Timedelta(days=pre_days)
        hi = anchor + pd.Timedelta(days=window_days)
        seg = price.loc[lo:hi].copy()
        if seg.empty:
            continue
        offset = (seg.index - anchor).days
        seg = pd.Series(seg.values, index=offset)
        seg = seg[~seg.index.duplicated(keep="first")]
        if rebase_at_anchor:
            at0 = seg.reindex([0]).dropna()
            base = at0.iloc[0] if not at0.empty else seg.iloc[0]
            seg = seg / base * 100.0
        out[label] = seg
    return pd.DataFrame(out).sort_index()
