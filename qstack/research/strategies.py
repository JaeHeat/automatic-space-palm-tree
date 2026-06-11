"""Strategy signal generators.

A strategy is just a callable ``(df) -> pd.Series`` returning the target position
for each bar in {-1, 0, +1}. Keeping the contract this small means strategies
compose freely and plug straight into :class:`qstack.research.Backtest` and the
workflow pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long when the fast SMA is above the slow SMA, flat otherwise."""
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float).fillna(0.0).rename("position")


def rsi_reversion(df: pd.DataFrame, period: int = 14, low: int = 30, high: int = 70) -> pd.Series:
    """Mean-reversion: go long oversold (<low), exit when overbought (>high)."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    pos = pd.Series(np.nan, index=df.index)
    pos[rsi < low] = 1.0
    pos[rsi > high] = 0.0
    return pos.ffill().fillna(0.0).rename("position")


def momentum(df: pd.DataFrame, lookback: int = 90) -> pd.Series:
    """Long when trailing return over ``lookback`` bars is positive."""
    ret = df["close"].pct_change(lookback)
    return (ret > 0).astype(float).fillna(0.0).rename("position")
