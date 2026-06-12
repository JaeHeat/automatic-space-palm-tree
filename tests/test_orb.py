"""Deterministic unit tests for the ORB engine, built on hand-crafted sessions."""

import pandas as pd
import pytest

from qstack.research import ORBParams, backtest_orb
from qstack.research.orb import Instrument, day_groups

# zero-cost contract so fills are exact and easy to reason about
T = Instrument("T", tick_size=0.25, point_value=1.0, commission_rt=0.0, slippage_ticks=0.0)


def _day(bars, date="2023-03-01"):
    """bars: list of (HH:MM, open, high, low, close) in ET -> OHLCV DataFrame."""
    idx = pd.to_datetime([f"{date} {t}" for t, *_ in bars]).tz_localize("America/New_York")
    data = [(o, h, l, c) for _, o, h, l, c in bars]
    return pd.DataFrame(data, index=idx, columns=["open", "high", "low", "close"])


# OR window 09:30-09:55 (or_minutes=30): high=100, low=98, range=2
OR_BARS = [("09:30", 99, 100, 98, 99), ("09:35", 99, 100, 98, 99), ("09:50", 99, 100, 98, 99)]
P = ORBParams(or_minutes=30, direction="both", stop_mult=1.0, target_mult=2.0)


def test_long_breakout_hits_target():
    df = _day(OR_BARS + [
        ("10:00", 100, 101, 99, 101),   # breakout above 100 -> long entry at 100
        ("10:05", 101, 104, 101, 104),  # target = 100 + 2*2 = 104 hit
    ])
    res = backtest_orb(df, P, T)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "long" and tr["entry"] == 100
    assert tr["reason"] == "target" and tr["exit"] == 104
    assert tr["pnl"] == pytest.approx(4.0)


def test_long_breakout_hits_stop():
    df = _day(OR_BARS + [
        ("10:00", 100, 101, 99, 101),   # long entry at 100
        ("10:05", 100, 100, 98, 98),    # stop = 100 - 1*2 = 98 hit
    ])
    res = backtest_orb(df, P, T)
    tr = res.trades.iloc[0]
    assert tr["reason"] == "stop" and tr["exit"] == 98
    assert tr["pnl"] == pytest.approx(-2.0)


def test_short_breakout():
    df = _day(OR_BARS + [
        ("10:00", 98, 99, 97, 97),      # breakdown below 98 -> short entry at 98
        ("10:05", 97, 97, 94, 94),      # target = 98 - 2*2 = 94 hit
    ])
    res = backtest_orb(df, P, T)
    tr = res.trades.iloc[0]
    assert tr["side"] == "short" and tr["entry"] == 98
    assert tr["reason"] == "target" and tr["pnl"] == pytest.approx(4.0)


def test_no_breakout_exits_at_eod():
    df = _day(OR_BARS + [
        ("10:00", 99, 99.5, 98.5, 99),  # never breaks the range
        ("15:55", 99, 99.5, 98.5, 99.5),
    ])
    res = backtest_orb(df, P, T)
    assert len(res.trades) == 0  # no breakout -> no trade


def test_breakout_then_eod_when_no_stop_or_target():
    df = _day(OR_BARS + [
        ("10:00", 100, 101, 99, 101),   # long entry at 100
        ("15:55", 101, 101.5, 100.5, 101),  # drifts, neither stop(98) nor target(104)
    ])
    res = backtest_orb(df, P, T)
    tr = res.trades.iloc[0]
    assert tr["reason"] == "eod" and tr["exit"] == 101
    assert tr["pnl"] == pytest.approx(1.0)


def test_groups_reuse_matches_direct():
    df = _day(OR_BARS + [("10:00", 100, 101, 99, 101), ("10:05", 101, 104, 101, 104)])
    direct = backtest_orb(df, P, T).stats
    viagroups = backtest_orb(df, P, T, groups=day_groups(df)).stats
    assert direct == viagroups


def test_direction_filter_blocks_shorts():
    df = _day(OR_BARS + [("10:00", 98, 99, 97, 97), ("10:05", 97, 97, 94, 94)])
    long_only = ORBParams(or_minutes=30, direction="long", stop_mult=1.0, target_mult=2.0)
    assert len(backtest_orb(df, long_only, T).trades) == 0  # short ignored
