"""Portfolio blend — combine de-correlated intraday strategies into one book.

No single strategy on NQ/ES hits win>60% AND PF>1.5 (see examples/orb_frontier.py
and intraday_lab.py). But blending strategies that profit in DIFFERENT regimes
smooths the combined equity, and at the portfolio level the natural unit is the
trading DAY: a diversified daily P&L can clear 60% green days and a daily profit
factor > 1.5 even when every component is weaker.

Components (each robust on its own, 1 lot):
  1. NQ ORB trend     -- breakout / momentum   (qstack ORB max_pf preset)
  2. NQ RSI(2) pullback -- mean-reversion in trend
  3. ES ORB trend     -- cross-instrument momentum

    python examples/portfolio_blend.py --nq nq_5m.csv --es es_5m.csv --plot blend.png
"""

from __future__ import annotations

import argparse
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


def day_stats(d: pd.Series, label: str) -> dict:
    if len(d) == 0:
        return {}
    up, dn = d[d > 0], d[d < 0]
    eq = d.cumsum()
    return {
        "book": label,
        "days": len(d),
        "green%": round((d > 0).mean() * 100, 1),
        "dayPF": round(up.sum() / -dn.sum(), 2) if dn.sum() != 0 else np.inf,
        "sharpe": round(d.mean() / d.std() * np.sqrt(252), 2) if d.std() > 0 else 0.0,
        "net$": round(d.sum()),
        "maxDD$": round((eq - eq.cummax()).min()),
        "avgDay$": round(d.mean()),
    }


def build_components(nq, es) -> dict[str, pd.Series]:
    comps = {}
    comps["NQ_ORB"] = daily(backtest_orb(nq, NQ_PRESETS["max_pf"], NQ).trades)
    comps["NQ_RSI"] = daily(backtest_rsi_pullback(nq, NQ))
    if es is not None:
        comps["ES_ORB"] = daily(backtest_orb(es, NQ_PRESETS["balanced"], ES).trades)
    return comps


def risk_weight(comps: dict[str, pd.Series], target_std=1000.0) -> dict[str, pd.Series]:
    """Scale each component to the same daily $ volatility (risk parity)."""
    return {k: v * (target_std / v.std()) for k, v in comps.items() if v.std() > 0}


def report(comps, split):
    idx = sorted(set().union(*[set(v.index) for v in comps.values()]))
    M = pd.DataFrame({k: v.reindex(idx).fillna(0.0) for k, v in comps.items()})
    sd = pd.Timestamp(split)

    print("=== components (1 lot each, daily) ===")
    rows = [day_stats(M[c], c) for c in M.columns]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== daily P&L correlation (diversification) ===")
    print(M.corr().round(2).to_string())

    # equal-risk blend
    rw = risk_weight({c: M[c] for c in M.columns})
    blend = pd.DataFrame(rw).sum(axis=1)
    print("\n=== BLEND (risk-parity, equal daily vol) ===")
    full = day_stats(blend, "blend FULL")
    tr = day_stats(blend[blend.index < sd], "blend train")
    te = day_stats(blend[blend.index >= sd], "blend test")
    print(pd.DataFrame([full, tr, te]).to_string(index=False))
    # weekly view (even smoother)
    wk = blend.resample("W").sum()
    print(f"\nweekly: green-weeks={ (wk>0).mean()*100:.1f}%  "
          f"weekPF={wk[wk>0].sum()/-wk[wk<0].sum():.2f}  weeks={len(wk)}")
    return M, blend, rw


def plot(M, blend, rw, split, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for c, s in rw.items():
        ax.plot(s.index, s.cumsum().values, lw=1.0, alpha=0.6, label=f"{c} (scaled)")
    ax.plot(blend.index, blend.cumsum().values, lw=2.2, color="black", label="BLEND")
    ax.axvline(pd.Timestamp(split), color="red", ls="--", lw=1, label="train/test")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("qstack portfolio blend — risk-parity daily equity (components vs blend)")
    ax.set_ylabel("cumulative P&L (risk-scaled $)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
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
    comps = build_components(nq, es)
    M, blend, rw = report(comps, args.split)
    if args.plot:
        plot(M, blend, rw, args.split, args.plot)


if __name__ == "__main__":
    main()
