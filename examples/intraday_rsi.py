"""Intraday RSI(2) pullback — buy oversold dips inside the trend (Connors-style).

A genuinely different strategy from ORB, found while searching for a high
win-rate edge on NQ. On 5-minute bars it goes long when RSI(2) is oversold and
price is above a trend MA (and optionally shorts the mirror in a downtrend),
exits when RSI recovers, and flats by the session close.

It is the best HIGH-WIN-RATE strategy I found on NQ: ~62-64% winners, profitable
out-of-sample. Its profit factor, though, sits around ~1.1 -- see the project
notes: on NQ intraday, win>60% AND PF>1.5 together appears unreachable (every
high-win strategy lands at PF ~1.0-1.3; every PF>1.5 strategy wins <60%).

    python examples/intraday_rsi.py --csv nq_5m.csv --plot rsi.png
"""

from __future__ import annotations

import argparse
from datetime import time

import numpy as np
import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import NQ
from qstack.research.orb import Instrument, trade_stats


def rsi(series: pd.Series, n: int) -> np.ndarray:
    d = series.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()


def backtest_rsi_pullback(df: pd.DataFrame, inst: Instrument = NQ, rsi_n: int = 2,
                          entry_th: float = 15, exit_th: float = 80, trend_ma: int = 100,
                          both: bool = True, session=(time(9, 30), time(16, 0))) -> pd.DataFrame:
    """Long oversold dips above the trend MA (and mirror shorts), exit on RSI recovery / EOD."""
    c = df["close"].to_numpy(); o = df["open"].to_numpy()
    cs = pd.Series(c)
    r = rsi(cs, rsi_n)
    tma = cs.rolling(trend_ma).mean().to_numpy()
    idx = df.index
    mins = (idx.hour * 60 + idx.minute).to_numpy()
    date = np.array(idx.date)
    rth = (mins >= session[0].hour * 60 + session[0].minute) & (mins < session[1].hour * 60 + session[1].minute)
    slip, pv, com = inst.slippage_pts, inst.point_value, inst.commission_rt

    out = []
    for d in np.unique(date):
        sel = np.where((date == d) & rth)[0]
        if len(sel) < 5:
            continue
        in_pos = False; side = 0; entry = 0.0; et = None
        for k in range(len(sel) - 1):
            i = sel[k]
            if not in_pos:
                if r[i] != r[i] or tma[i] != tma[i]:
                    continue
                longs = r[i] < entry_th and c[i] > tma[i]
                shorts = both and r[i] > (100 - entry_th) and c[i] < tma[i]
                if longs or shorts:
                    j = sel[k + 1]; side = 1 if longs else -1
                    entry = o[j] + side * slip; et = idx[j]; in_pos = True
            else:
                recovered = (r[sel[k]] > exit_th) if side == 1 else (r[sel[k]] < 100 - exit_th)
                if recovered:
                    j = sel[k + 1]
                    ex = o[j] - side * slip
                    out.append((d, side, et, idx[j], (ex - entry) * side * pv - com)); in_pos = False
        if in_pos:
            i2 = sel[-1]; ex = c[i2] - side * slip
            out.append((d, side, et, idx[i2], (ex - entry) * side * pv - com))
    return pd.DataFrame(out, columns=["date", "side", "entry_time", "exit_time", "pnl"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    df = load_ohlcv_csv(args.csv)
    t = backtest_rsi_pullback(df)
    sd = pd.Timestamp(args.split).date()
    nd = t["date"].nunique()
    for lab, sub in [("FULL 2022-26", t), ("IN-SAMPLE", t[t.date < sd]), ("OUT-OF-SAMPLE", t[t.date >= sd])]:
        s = trade_stats(sub.assign(date=sub["date"]), sub["date"].nunique() or 1)
        print(f"{lab:16s} N={s['n_trades']:5d} win={s['win_rate']*100:5.1f}% "
              f"PF={s['profit_factor']:.2f} net=${s['net_pnl']:>9,.0f} avg=${s['avg_trade']:.0f}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        eq = t.sort_values("exit_time").set_index("exit_time")["pnl"].cumsum()
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(eq.index, eq.values, lw=1.3)
        ax.axvline(pd.Timestamp(args.split, tz=df.index.tz), color="red", ls="--", lw=1, label="train/test split")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("qstack RSI(2) pullback — NQ 1-lot cumulative P&L (~63% win, PF ~1.1)")
        ax.set_ylabel("cumulative net P&L ($)"); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=120)
        print(f"saved -> {args.plot}")


if __name__ == "__main__":
    main()
