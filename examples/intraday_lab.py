"""Intraday strategy lab — search win-rate / profit-factor across the design space.

Consolidates the exploration done in pursuit of a "win>60% AND PF>1.5" intraday
strategy on index futures. From one 5-minute CSV it can resample to higher
timeframes and sweep:

  * timeframe        : 5 / 10 / 15 / 30 minute bars (resampled from the 5m input)
  * trading window   : full RTH or a "kill zone" (NY open, silver-bullet, NY PM…)
  * entry style      : trend pullback (RSI-2), range breakout, or RSI mean-revert
  * HTF confluence   : align with a higher-timeframe EMA trend
  * risk model       : ATR-based stop and R-multiple target (or RSI-recovery exit)

For every config it reports win rate and profit factor on the full sample and on
a train/test split, and flags anything clearing the targets.

    python examples/intraday_lab.py --csv nq_5m.csv --instrument NQ

NB on data: a true ORB / sub-5m search needs 1-minute bars; this tool reads
whatever bar size you give it, so point it at a 1m CSV to extend the search
below 5m.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import ES, NQ
from qstack.research.orb import Instrument, trade_stats

INSTRUMENTS = {"NQ": NQ, "ES": ES}

# Kill zones / windows in ET minutes-from-midnight.
WINDOWS = {
    "RTH_0930_1600": (570, 960),
    "NYAM_0930_1130": (570, 690),
    "SB_AM_1000_1100": (600, 660),
    "NYPM_1330_1500": (810, 900),
    "OPEN_0930_1030": (570, 630),
}


def resample(df, tf):
    if tf == 5:
        return df
    return (df.resample(f"{tf}min", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna())


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df, n):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


class Lab:
    """Holds the 5m data and caches per-timeframe arrays for fast sweeping."""

    def __init__(self, df5, inst: Instrument):
        self.df5 = df5
        self.inst = inst
        self._tf, self._htf = {}, {}

    def tf(self, tf):
        if tf not in self._tf:
            df = resample(self.df5, tf)
            self._tf[tf] = {
                "df": df, "o": df["open"].to_numpy(), "h": df["high"].to_numpy(),
                "l": df["low"].to_numpy(), "c": df["close"].to_numpy(),
                "rs": _rsi(df["close"], 2).to_numpy(), "at": _atr(df, 14).to_numpy(),
                "hh": df["high"].rolling(12).max().shift(1).to_numpy(),
                "ll": df["low"].rolling(12).min().shift(1).to_numpy(),
                "mins": (df.index.hour * 60 + df.index.minute).to_numpy(),
                "date": np.array(df.index.date),
            }
            d = self._tf[tf]
            days = [(x, np.where(d["date"] == x)[0]) for x in np.unique(d["date"])]
            d["days"] = [(x, s) for x, s in days if len(s) >= 5]
        return self._tf[tf]

    def htf_up(self, tf, htf, ma):
        key = (tf, htf, ma)
        if key not in self._htf:
            r = resample(self.df5, htf)
            up = (r["close"] > _ema(r["close"], ma)).shift(1)
            self._htf[key] = up.reindex(self.tf(tf)["df"].index, method="ffill").astype(float).to_numpy()
        return self._htf[key]

    def run(self, tf, entry, win, htf, ma, rsi_th, stop_atr, rr, both):
        D = self.tf(tf)
        up = self.htf_up(tf, htf, ma)
        o, h, l, c, rs, at, hh, ll, mins = (D["o"], D["h"], D["l"], D["c"], D["rs"],
                                            D["at"], D["hh"], D["ll"], D["mins"])
        w0, w1 = win
        inwin = (mins >= w0) & (mins < w1)
        upb, downb = up > 0.5, (up < 0.5) & ~np.isnan(up)
        if entry == "pullback":
            longs = upb & (rs < rsi_th) & (c > o)
            shorts = downb & (rs > 100 - rsi_th) & (c < o)
        elif entry == "breakout":
            longs, shorts = upb & (c > hh), downb & (c < ll)
        else:  # rsi mean-reversion (no trend filter)
            longs = (rs < rsi_th) & (c > o)
            shorts = both & (rs > 100 - rsi_th) & (c < o)
        valid = inwin & ~np.isnan(at)
        longs, shorts = longs & valid, shorts & valid
        pv, com, slip = self.inst.point_value, self.inst.commission_rt, self.inst.slippage_pts

        out = []
        for _, sel in D["days"]:
            n = len(sel); k = 0
            while k < n - 1:
                i = sel[k]
                side = 1 if longs[i] else (-1 if shorts[i] else 0)
                if side == 0:
                    k += 1; continue
                j = sel[k + 1]; entry_px = o[j] + side * slip; risk = stop_atr * at[i]
                if risk <= 0:
                    k += 1; continue
                stp, tgt = entry_px - side * risk, entry_px + side * rr * risk
                ex = None; m = k + 1
                while m < n:
                    i2 = sel[m]
                    if side == 1:
                        if l[i2] <= stp: ex = min(stp, o[i2]) - slip
                        elif h[i2] >= tgt: ex = max(tgt, o[i2]) - slip
                    else:
                        if h[i2] >= stp: ex = max(stp, o[i2]) + slip
                        elif l[i2] <= tgt: ex = min(tgt, o[i2]) + slip
                    if ex is not None:
                        break
                    m += 1
                if ex is None:
                    i2 = sel[-1]; ex = c[i2] - side * slip; m = n - 1
                out.append((D["date"][i], (ex - entry_px) * side * pv - com))
                k = m + 1
        return pd.DataFrame(out, columns=["date", "pnl"])


def sweep(lab: Lab, split, timeframes):
    grid = {
        "tf": timeframes, "entry": ["pullback", "breakout"], "win": list(WINDOWS),
        "htf": [60], "ma": [20, 50], "rsi_th": [35], "stop_atr": [1.0, 1.5, 2.0],
        "rr": [1.0, 1.5, 2.0, 2.5], "both": [True],
    }
    sd = pd.Timestamp(split).date()
    rows = []
    for combo in itertools.product(*grid.values()):
        p = dict(zip(grid, combo))
        t = lab.run(p["tf"], p["entry"], WINDOWS[p["win"]], p["htf"], p["ma"],
                    p["rsi_th"], p["stop_atr"], p["rr"], p["both"])
        if len(t) < 150:
            continue
        s = trade_stats(t, t["date"].nunique())
        tr = trade_stats(t[t.date < sd], max(t[t.date < sd]["date"].nunique(), 1))
        te = trade_stats(t[t.date >= sd], max(t[t.date >= sd]["date"].nunique(), 1))
        rows.append({**{k: p[k] for k in ("tf", "entry", "win", "ma", "stop_atr", "rr")},
                     "N": s["n_trades"], "win": round(s["win_rate"] * 100, 1),
                     "PF": round(s["profit_factor"], 2), "netK": round(s["net_pnl"] / 1000),
                     "tr_win": round(tr["win_rate"] * 100, 1), "tr_pf": round(tr["profit_factor"], 2),
                     "te_win": round(te["win_rate"] * 100, 1), "te_pf": round(te["profit_factor"], 2)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--instrument", default="NQ", choices=sorted(INSTRUMENTS))
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--timeframes", default="5,15,30")
    args = ap.parse_args(argv)

    tfs = [int(x) for x in args.timeframes.split(",")]
    lab = Lab(load_ohlcv_csv(args.csv), INSTRUMENTS[args.instrument])
    R = sweep(lab, args.split, tfs)
    print(f"{args.instrument}: {len(R)} configs | best win={R['win'].max()}%  best PF={R['PF'].max()}")
    hit = R[(R["win"] > 60) & (R["PF"] > 1.5)]
    print(f"configs meeting win>60 AND PF>1.5: {len(hit)}")
    if len(hit):
        print(hit.sort_values("PF", ascending=False).head(20).to_string(index=False))
    print("\nTop 12 by PF (win>=55):")
    print(R[R["win"] >= 55].sort_values("PF", ascending=False).head(12).to_string(index=False))
    print("\nTop 12 by win rate (PF>=1.2):")
    print(R[R["PF"] >= 1.2].sort_values("win", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
