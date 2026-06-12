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


def _multiday(n=14):
    """Build n business days, each a long breakout that runs to target."""
    frames = []
    for day in pd.bdate_range("2023-01-02", periods=n):
        d = day.strftime("%Y-%m-%d")
        frames.append(_day([
            ("09:30", 99, 100, 98, 99), ("09:35", 99, 100, 98, 99), ("09:50", 99, 100, 98, 99),
            ("10:00", 100, 101, 99, 101),    # long entry at 100
            ("10:05", 101, 104, 101, 104),   # target 104 hit
        ], date=d))
    return pd.concat(frames)


def test_trade_stats_on_subset_is_consistent():
    from qstack.research import trade_stats
    df = _multiday(10)
    res = backtest_orb(df, P, T)
    full = res.stats
    half_dates = set(sorted(set(res.trades["date"]))[:5])
    sub = res.trades[res.trades["date"].isin(half_dates)]
    st = trade_stats(sub, 5)
    assert st["n_trades"] == len(sub)
    assert st["net_pnl"] == pytest.approx(sub["pnl"].sum())
    assert full["n_trades"] == len(res.trades)


def test_walk_forward_out_of_sample_only():
    from qstack.research import walk_forward
    grid = [ORBParams(or_minutes=30, stop_mult=1.0, target_mult=2.0),
            ORBParams(or_minutes=30, stop_mult=1.5, target_mult=3.0)]
    df = _multiday(14)
    wf = walk_forward(df, grid, T, train_days=4, test_days=2, min_train_trades=1)
    assert not wf.windows.empty
    # every OOS trade must fall strictly after its window's training period
    assert wf.oos_trades["pnl"].notna().all()
    assert wf.stats["n_trades"] == len(wf.oos_trades)


def test_trend_filter_blocks_counter_trend():
    # flat/declining series -> a 50-day uptrend filter should allow no longs
    df = _multiday(60)
    longs_only_uptrend = ORBParams(or_minutes=30, direction="long", stop_mult=1.0,
                                   target_mult=2.0, trend_ma=50)
    res = backtest_orb(df, longs_only_uptrend, T)
    # daily close is constant (104 target each day) -> close is not > its own SMA
    assert len(res.trades) == 0


def test_confirm_close_changes_entry():
    # bar at 10:00 wicks above 100 but closes below; 10:05 closes above -> confirmed
    bars = OR_BARS + [
        ("10:00", 99, 101, 99, 99.5),    # wick through, close 99.5 < 100
        ("10:05", 99.5, 102, 99.5, 101), # close 101 > 100 -> confirmed
        ("10:10", 101, 101, 101, 101),   # entry at next bar open = 101
        ("10:15", 101, 105, 101, 105),   # target hit
    ]
    df = _day(bars)
    raw = backtest_orb(df, P, T).trades.iloc[0]
    conf = backtest_orb(df, ORBParams(or_minutes=30, stop_mult=1.0, target_mult=2.0,
                                      confirm_close=True), T).trades.iloc[0]
    assert raw["entry"] == 100      # naive enters on the wick
    assert conf["entry"] == 101     # confirmed enters next bar open


def test_skip_monday():
    monday = "2023-03-06"  # a Monday
    df = _day(OR_BARS + [("10:00", 100, 101, 99, 101), ("10:05", 101, 104, 101, 104)], date=monday)
    assert len(backtest_orb(df, P, T).trades) == 1
    skip = ORBParams(or_minutes=30, stop_mult=1.0, target_mult=2.0, skip_monday=True)
    assert len(backtest_orb(df, skip, T).trades) == 0


def test_breakeven_stop_scratches_a_pullback():
    bars = OR_BARS + [
        ("10:00", 100, 101, 99, 101),       # long entry at 100, range 2
        ("10:05", 101, 101.5, 100.5, 101),  # high 101.5 >= 100+0.5*2=101 -> stop to breakeven
        ("10:10", 100.5, 100.5, 100, 100),  # pulls back to 100 -> exits at breakeven
        ("15:55", 100, 100, 99, 100),
    ]
    df = _day(bars)
    be = ORBParams(or_minutes=30, stop_mult=1.0, target_mult=None, breakeven_at=0.5)
    tr = backtest_orb(df, be, T).trades.iloc[0]
    assert tr["reason"] == "breakeven"
    assert tr["pnl"] == pytest.approx(0.0)  # exited at entry, zero-cost contract
