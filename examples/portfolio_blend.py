"""Portfolio blend — combine de-correlated intraday strategies into one book.

No single strategy on NQ/ES hits win>60% AND PF>1.5 at the trade level (see
examples/orb_frontier.py and intraday_lab.py). But blending strategies that
profit in DIFFERENT regimes smooths the combined equity, and at the portfolio
level the natural unit is the trading DAY (or week): a diversified book clears
the targets there even when every component is weaker.

Four near-uncorrelated sleeves (1 lot each, then risk-weighted):
  1. NQ_ORB     -- ORB breakout, wide target   (max_pf preset)   momentum
  2. NQ_RSI     -- RSI(2) pullback in trend                      mean-reversion
  3. ES_ORB     -- ORB breakout on ES           (balanced preset) momentum
  4. NQ_ORBwin  -- ORB breakout, tight target  (max_win preset)  high win-rate

Weights are optimized on TRAIN only (max daily Sharpe) and reported on the
held-out TEST period, alongside an un-fitted equal-risk baseline so you can see
the optimization generalizes rather than overfits.

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
    """Collapse a trade list to one P&L number per calendar trading day."""
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


def metrics(x: pd.Series) -> dict:
    if len(x) == 0 or x.std() == 0:
        return {"green%": 0, "dayPF": 0, "sharpe": 0, "net$": 0, "maxDD$": 0}
    up, dn = x[x > 0].sum(), -x[x < 0].sum()
    eq = x.cumsum()
    return {"green%": round((x > 0).mean() * 100, 1),
            "dayPF": round(up / dn, 2) if dn > 0 else np.inf,
            "sharpe": round(x.mean() / x.std() * np.sqrt(252), 2),
            "net$": round(x.sum()), "maxDD$": round((eq - eq.cummax()).min())}


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


def report(comps, split):
    idx = sorted(set().union(*[set(v.index) for v in comps.values()]))
    M = pd.DataFrame({k: v.reindex(idx).fillna(0.0) for k, v in comps.items()})
    sd = pd.Timestamp(split)
    tr = M.index < sd

    print("=== components (1 lot each, daily) ===")
    print(pd.DataFrame({c: metrics(M[c]) for c in M.columns}).T.to_string())
    print("\n=== daily P&L correlation ===")
    print(M.corr().round(2).to_string())

    # risk-normalise each sleeve to unit daily vol using TRAIN std (no lookahead)
    N = M / M[tr].std()
    names = list(M.columns)
    eq_w = np.full(len(names), 1 / len(names))
    opt_w = optimize_weights(N[tr])

    for w, label in [(eq_w, "EQUAL-RISK (un-fitted baseline)"),
                     (opt_w, "TRAIN-OPTIMISED (max train Sharpe)")]:
        port = (N * w).sum(axis=1)
        gw_f, pw_f = weekly(port)
        gw_t, pw_t = weekly(port[~tr])
        print(f"\n=== {label} ===  weights={dict(zip(names, np.round(w, 2)))}")
        print(f"  FULL : {metrics(port)}   weekly green={gw_f}% PF={pw_f}")
        print(f"  TRAIN: {metrics(port[tr])}")
        print(f"  TEST : {metrics(port[~tr])}   weekly green={gw_t}% PF={pw_t}")
    return M, N, opt_w, names


def plot(N, w, names, split, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    blend = (N * w).sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(names):
        ax.plot(N.index, (N[c] * w[i]).cumsum().values, lw=1.0, alpha=0.55, label=f"{c} (w={w[i]:.2f})")
    ax.plot(blend.index, blend.cumsum().values, lw=2.4, color="black", label="BLEND")
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
    M, N, opt_w, names = report(build_components(nq, es), args.split)
    if args.plot:
        plot(N, opt_w, names, args.split, args.plot)


if __name__ == "__main__":
    main()
