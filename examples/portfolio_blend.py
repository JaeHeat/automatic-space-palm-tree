"""Portfolio blend — combine de-correlated intraday strategies into one book.

No single strategy on NQ/ES hits win>60% AND PF>1.5 at the trade level (see
examples/orb_frontier.py and intraday_lab.py). But blending strategies that
profit in DIFFERENT regimes smooths the combined equity, and at the portfolio
level the natural unit is the trading DAY/WEEK, where a diversified book clears
the targets even when every component is weaker.

Four near-uncorrelated sleeves (1 lot each, then risk-weighted):
  1. NQ_ORB     -- ORB breakout, wide target   (max_pf preset)   momentum
  2. NQ_RSI     -- RSI(2) pullback in trend                      mean-reversion
  3. NQ_ORBwin  -- ORB breakout, tight target  (max_win preset)  high win-rate
  4. ES_ORB     -- ORB breakout on ES           (balanced preset) momentum

This reports: fixed equal-risk and train-optimised weights, a WALK-FORWARD
(quarterly-refit) weighting, the real-$ max drawdown of a 1-lot book, and an
equity-throttle risk overlay that cuts the drawdown.

    python examples/portfolio_blend.py --nq nq_5m.csv --es es_5m.csv --plot blend.png
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

from qstack.connect import load_ohlcv_csv
from qstack.research import ES, NQ, backtest_orb
from qstack.research.orb import NQ_PRESETS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intraday_rsi import backtest_rsi_pullback  # noqa: E402


def daily(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    s = trades.groupby("date")["pnl"].sum()
    s.index = pd.to_datetime(list(s.index))
    return s.sort_index()


def build_components(nq, es) -> dict[str, pd.Series]:
    comps = {
        "NQ_ORB": daily(backtest_orb(nq, NQ_PRESETS["max_pf"], NQ).trades),
        "NQ_RSI": daily(backtest_rsi_pullback(nq, NQ)),
        "NQ_ORBwin": daily(backtest_orb(nq, NQ_PRESETS["max_win"], NQ).trades),
    }
    if es is not None:
        comps["ES_ORB"] = daily(backtest_orb(es, NQ_PRESETS["balanced"], ES).trades)
    return comps


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Worst peak-to-trough drop and its underwater duration in calendar days."""
    dd = equity - equity.cummax()
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    after = equity.loc[trough:]
    rec = after[after >= equity.loc[peak]]
    recovered = rec.index[0] if len(rec) else equity.index[-1]
    return dd.min(), (recovered - peak).days


def metrics(x: pd.Series) -> dict:
    if len(x) == 0 or x.std() == 0:
        return {"green%": 0, "dayPF": 0, "sharpe": 0, "net$": 0, "maxDD$": 0, "DDdays": 0, "calmar": 0}
    up, dn = x[x > 0].sum(), -x[x < 0].sum()
    mdd, ddur = max_drawdown(x.cumsum())
    return {"green%": round((x > 0).mean() * 100, 1),
            "dayPF": round(up / dn, 2) if dn > 0 else np.inf,
            "sharpe": round(x.mean() / x.std() * np.sqrt(252), 2),
            "net$": round(x.sum()), "maxDD$": round(mdd), "DDdays": ddur,
            "calmar": round((x.mean() * 252) / abs(mdd), 2) if mdd != 0 else np.inf}


def weekly(x: pd.Series) -> tuple[float, float]:
    w = x.resample("W").sum()
    up, dn = w[w > 0].sum(), -w[w < 0].sum()
    return round((w > 0).mean() * 100, 1), round(up / dn, 2) if dn > 0 else np.inf


def optimize_weights(N_train: pd.DataFrame, step: int = 10) -> np.ndarray:
    """Grid-search long-only weights (sum=1) maximising TRAIN daily Sharpe."""
    k = N_train.shape[1]
    best_w, best_s = None, -np.inf
    for comp in itertools.product(range(step + 1), repeat=k):
        if sum(comp) != step:
            continue
        w = np.array(comp) / step
        port = (N_train * w).sum(axis=1)
        s = port.mean() / port.std() * np.sqrt(252) if port.std() > 0 else -np.inf
        if s > best_s:
            best_s, best_w = s, w
    return best_w


def walk_forward_weights(M: pd.DataFrame, train: int = 252, test: int = 63) -> pd.Series:
    """Re-optimise weights each quarter on the trailing year; trade them next quarter."""
    oos, t = [], train
    while t < len(M):
        tr, te = M.iloc[t - train:t], M.iloc[t:t + test]
        sd = tr.std().replace(0, 1)
        w = optimize_weights(tr / sd)
        oos.append((te / sd * w).sum(axis=1))
        t += test
    return pd.concat(oos) if oos else pd.Series(dtype=float)


def equity_throttle(x: pd.Series, window: int = 40, factor: float = 0.5) -> pd.Series:
    """Halve exposure after the book's trailing-window P&L turns negative (lagged)."""
    roll = x.rolling(window).sum().shift(1)
    return x * np.where(roll < 0, factor, 1.0)


def report(comps, split):
    idx = sorted(set().union(*[set(v.index) for v in comps.values()]))
    M = pd.DataFrame({k: v.reindex(idx).fillna(0.0) for k, v in comps.items()})
    sd = pd.Timestamp(split)
    tr = M.index < sd
    names = list(M.columns)

    print("=== components (1 lot each, daily) ===")
    print(pd.DataFrame({c: metrics(M[c]) for c in names}).T.to_string())
    print("\n=== daily P&L correlation (all days / worst-decile days) ===")
    N = M / M[tr].std()
    eqr = N.mean(axis=1)
    worst = eqr < eqr.quantile(0.10)
    print("all:\n" + M.corr().round(2).to_string())
    print("worst-decile (tail co-movement):\n" + M[worst].corr().round(2).to_string())

    eq_w = np.full(len(names), 1 / len(names))
    opt_w = optimize_weights(N[tr])
    for w, label in [(eq_w, "EQUAL-RISK (un-fitted)"), (opt_w, "TRAIN-OPTIMISED")]:
        port = (N * w).sum(axis=1)
        gwf, pwf = weekly(port)
        gwt, pwt = weekly(port[~tr])
        print(f"\n=== {label} === weights={dict(zip(names, np.round(w, 2)))}")
        print(f"  FULL : {metrics(port)}  weekly {gwf}%/{pwf}")
        print(f"  TEST : {metrics(port[~tr])}  weekly {gwt}%/{pwt}")

    wf = walk_forward_weights(M)
    print(f"\n=== WALK-FORWARD weights (quarterly refit) ===\n  OOS: {metrics(wf)}  weekly {weekly(wf)}")

    real = M[[c for c in names if c != 'ES_ORB']].sum(axis=1)  # tradeable 1-lot book, ES dropped
    print(f"\n=== REAL $ (1 lot each: {[c for c in names if c!='ES_ORB']}) ===")
    print(f"  no overlay : {metrics(real)}")
    print(f"  +throttle  : {metrics(equity_throttle(real))}")
    return M, N, eq_w, names


def plot(N, w, names, split, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    blend = (N * w).sum(axis=1)
    thr = equity_throttle(blend)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(names):
        ax.plot(N.index, (N[c] * w[i]).cumsum().values, lw=0.9, alpha=0.5, label=f"{c} (w={w[i]:.2f})")
    ax.plot(blend.index, blend.cumsum().values, lw=2.2, color="black", label="BLEND (equal-risk)")
    ax.plot(thr.index, thr.cumsum().values, lw=1.6, color="darkgreen", ls="--", label="BLEND + equity throttle")
    ax.axvline(pd.Timestamp(split), color="red", ls="--", lw=1, label="train/test")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("qstack portfolio blend — risk-weighted daily equity (4 sleeves)")
    ax.set_ylabel("cumulative P&L (risk-scaled $)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nsaved -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nq", required=True)
    ap.add_argument("--es", default=None)
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    nq = load_ohlcv_csv(args.nq)
    es = load_ohlcv_csv(args.es) if args.es else None
    M, N, w, names = report(build_components(nq, es), args.split)
    if args.plot:
        plot(N, w, names, args.split, args.plot)


if __name__ == "__main__":
    main()
