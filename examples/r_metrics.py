"""Express the blend and its sleeves in R (risk multiples), not dollars.

1R = the initial risk taken on a trade (the planned stop distance). Reporting in
R is account-size- and instrument-agnostic: a +2R day means you made twice what
you risked, regardless of contract or capital.

Per-sleeve risk unit:
  * ORB sleeves : 1R = stop_mult x opening-range x point_value  (an exact stop)
  * RSI sleeve  : has no hard stop, so 1R := 1 x ATR(14) at entry (the natural
                  stop you would have used) -- documented assumption.

What R reveals that dollars hide: the RSI sleeve is a tail-risk sleeve (average
loss ~1.7R, single days at -20R/-38R), because with no stop its losers run to
large multiples of the risk unit. The pure-ORB sleeves are clean (max DD ~5-7R).

    python examples/r_metrics.py --nq nq_5m.csv --es es_5m.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import ES, NQ, backtest_orb
from qstack.research.orb import Instrument, NQ_PRESETS


def orb_R(df, preset: str, inst: Instrument) -> pd.DataFrame:
    p = NQ_PRESETS[preset]
    t = backtest_orb(df, p, inst).trades.copy()
    risk = p.stop_mult * t["or_range"] * inst.point_value          # $ risked per lot = 1R
    t["R"] = t["pnl"] / risk
    t["date"] = pd.to_datetime(t["date"])
    return t[["date", "R"]]


def rsi_R(df, inst: Instrument, rsi_n=2, entry_th=15, exit_th=80, trend_ma=100,
         stop_atr: float | None = None) -> pd.DataFrame:
    """RSI(2) pullback expressed in R.

    With ``stop_atr`` set, a hard stop at ``stop_atr x ATR`` caps each loss and
    1R = the stop distance. With ``stop_atr=None`` there is no stop and
    1R = 1 x ATR(14) at entry (the natural stop you would have used).
    """
    c = df["close"].to_numpy(); o = df["open"].to_numpy()
    h = df["high"].to_numpy(); l = df["low"].to_numpy()
    cs = pd.Series(c)
    d = cs.diff()
    up = d.clip(lower=0).rolling(rsi_n).mean(); dn = (-d.clip(upper=0)).rolling(rsi_n).mean()
    r = (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()
    e = cs.ewm(span=trend_ma, adjust=False).mean().to_numpy()
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    at = tr.rolling(14).mean().to_numpy()
    idx = df.index; mins = (idx.hour * 60 + idx.minute).to_numpy(); date = np.array(idx.date)
    rth = (mins >= 570) & (mins < 960)
    pv, com, slip = inst.point_value, inst.commission_rt, inst.slippage_pts
    Rden = stop_atr if stop_atr else 1.0      # 1R denominator in ATRs

    out = []
    for dd in np.unique(date):
        sel = np.where((date == dd) & rth)[0]
        if len(sel) < 5:
            continue
        inpos = False; side = 0; entry = 0.0; atr_e = 0.0; stop = None
        for k in range(len(sel) - 1):
            i = sel[k]
            if not inpos:
                if r[i] != r[i] or e[i] != e[i] or at[i] != at[i] or at[i] <= 0:
                    continue
                longs = r[i] < entry_th and c[i] > e[i]
                shorts = r[i] > 100 - entry_th and c[i] < e[i]
                if longs or shorts:
                    j = sel[k + 1]; side = 1 if longs else -1
                    entry = o[j] + side * slip; atr_e = at[i]
                    stop = (entry - side * stop_atr * atr_e) if stop_atr else None
                    inpos = True
            else:
                i2 = sel[k]; ex = None
                if stop is not None:
                    if side == 1 and l[i2] <= stop:
                        ex = min(stop, o[i2]) - slip
                    elif side == -1 and h[i2] >= stop:
                        ex = max(stop, o[i2]) + slip
                if ex is None:
                    rec = (r[i2] > exit_th) if side == 1 else (r[i2] < 100 - exit_th)
                    if rec:
                        j = sel[k + 1] if k + 1 < len(sel) else i2
                        ex = o[j] - side * slip
                if ex is not None:
                    out.append((dd, ((ex - entry) * side * pv - com) / (Rden * atr_e * pv))); inpos = False
        if inpos:
            ex = c[sel[-1]] - side * slip
            out.append((dd, ((ex - entry) * side * pv - com) / (Rden * atr_e * pv)))
    t = pd.DataFrame(out, columns=["date", "R"]); t["date"] = pd.to_datetime(t["date"])
    return t


def maxdd_R(equity: pd.Series) -> float:
    return (equity - equity.cummax()).min()


def per_sleeve(t: pd.DataFrame, name: str) -> dict:
    R = t["R"]; w = R[R > 0]; l = R[R < 0]
    return {"sleeve": name, "trades": len(R), "win%": round((R > 0).mean() * 100, 1),
            "exp R/trade": round(R.mean(), 3), "avgWin R": round(w.mean(), 2),
            "avgLoss R": round(l.mean(), 2), "payoff": round(w.mean() / -l.mean(), 2),
            "total R": round(R.sum(), 1), "maxDD R": round(maxdd_R(R.cumsum()), 1)}


def blend_stats(x: pd.Series, label: str):
    eq = x.cumsum(); mdd = maxdd_R(eq)
    years = (x.index[-1] - x.index[0]).days / 365
    wk = x.resample("W").sum()
    up, dn = x[x > 0].sum(), -x[x < 0].sum()
    print(f"\n{label}")
    print(f"  net={x.sum():.0f}R   maxDD={mdd:.1f}R   {x.sum()/years:.0f}R/yr   "
          f"MAR={(x.sum()/years)/abs(mdd):.2f}   sharpe={x.mean()/x.std()*np.sqrt(252):.2f}")
    print(f"  green={ (x>0).mean()*100:.1f}%  dayPF={up/dn:.2f}  "
          f"weekly green={ (wk>0).mean()*100:.1f}%  weekly PF={wk[wk>0].sum()/-wk[wk<0].sum():.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nq", required=True)
    ap.add_argument("--es", default=None)
    ap.add_argument("--rsi-stop", type=float, default=None,
                    help="hard ATR stop on the RSI sleeve (e.g. 3.0); default = no stop")
    args = ap.parse_args(argv)

    nq = load_ohlcv_csv(args.nq)
    es = load_ohlcv_csv(args.es) if args.es else None
    sleeves = {"NQ_ORB": orb_R(nq, "max_pf", NQ),
               "NQ_RSI": rsi_R(nq, NQ, stop_atr=args.rsi_stop),
               "NQ_ORBwin": orb_R(nq, "max_win", NQ)}
    if es is not None:
        sleeves["ES_ORB"] = orb_R(es, "balanced", ES)

    print("=== PER SLEEVE (1R = initial stop risk per trade) ===")
    print(pd.DataFrame([per_sleeve(t, n) for n, t in sleeves.items()]).to_string(index=False))

    dser = {n: t.groupby("date")["R"].sum() for n, t in sleeves.items()}
    idx = sorted(set().union(*[set(s.index) for s in dser.values()]))
    M = pd.DataFrame({n: dser[n].reindex(idx).fillna(0.0) for n in dser})

    blend_stats(M.sum(axis=1), "=== BLEND: 1R/trade per sleeve (natural sizing) ===")
    N = M / M.std()  # equal daily-R volatility
    blend_stats(N.sum(axis=1), "=== BLEND: equal daily-R-vol weighting (recommended) ===")
    orb_only = [n for n in M.columns if n != "NQ_RSI"]
    blend_stats((M[orb_only] / M[orb_only].std()).sum(axis=1),
                "=== BLEND: ORB-only (drop the tail-heavy RSI sleeve) ===")

    print("\nworst 5 days (R, per sleeve):")
    b = M.sum(axis=1)
    for d in b.nsmallest(5).index:
        print(f"  {d.date()} total={b[d]:+.1f}R | " + " ".join(f"{n}={M.loc[d, n]:+.1f}" for n in M.columns))


if __name__ == "__main__":
    main()
