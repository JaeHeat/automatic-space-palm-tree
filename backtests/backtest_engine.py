"""Shared backtest utilities: data loading, cost-aware PnL, performance metrics.

Conventions
-----------
* Daily crypto data -> annualization factor 365.
* A "position" series holds target weights known at the *close* of day t; it earns
  the asset return from t -> t+1. We therefore shift positions by 1 day before
  multiplying by returns to avoid look-ahead.
* Costs are charged on traded notional: cost_t = bps/1e4 * sum(|w_t - w_{t-1}|).
"""
import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ANN = 365  # crypto trades daily


def load_closes():
    """Return a DataFrame of daily close prices, columns = coin tickers."""
    series = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*_USD.csv"))):
        sym = os.path.basename(path).replace("_USD.csv", "")
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        series[sym] = df["close"]
    closes = pd.DataFrame(series).sort_index()
    return closes


def perf_metrics(returns, ann=ANN):
    """Compute performance stats from a daily strategy-return series (already net)."""
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std() == 0:
        return {k: float("nan") for k in
                ["total_return", "cagr", "vol", "sharpe", "sortino",
                 "max_drawdown", "calmar", "hit_rate", "n_days"]}
    equity = (1 + r).cumprod()
    years = len(r) / ann
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = (r.mean() / r.std()) * np.sqrt(ann) if r.std() else float("nan")
    downside = r[r < 0].std()
    sortino = (r.mean() / downside) * np.sqrt(ann) if downside else float("nan")
    dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd else float("nan")
    hit = (r > 0).mean()
    return {"total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "sortino": sortino, "max_drawdown": dd, "calmar": calmar,
            "hit_rate": hit, "n_days": len(r)}


def apply_costs(weights, asset_returns, cost_bps=10.0):
    """Turn a target-weight frame/series into a net daily return series.

    weights      : DataFrame [date x asset] or Series [date] of target weights
                   decided at close of day t.
    asset_returns: same shape, simple returns from t -> t+1.
    cost_bps     : round-trip-agnostic per-unit-traded-notional cost in bps.
    """
    w = weights.fillna(0.0)
    # position earns next-day return
    gross = (w.shift(1) * asset_returns)
    gross = gross.sum(axis=1) if isinstance(gross, pd.DataFrame) else gross
    turnover = (w - w.shift(1)).abs()
    turnover = turnover.sum(axis=1) if isinstance(turnover, pd.DataFrame) else turnover
    cost = turnover.shift(1) * (cost_bps / 1e4)
    net = gross - cost.fillna(0.0)
    return net, turnover


def fmt(metrics):
    """Pretty one-liner for a metrics dict."""
    return (f"ret={metrics['total_return']*100:7.1f}%  CAGR={metrics['cagr']*100:6.1f}%  "
            f"vol={metrics['vol']*100:5.1f}%  Sharpe={metrics['sharpe']:5.2f}  "
            f"Sortino={metrics['sortino']:5.2f}  maxDD={metrics['max_drawdown']*100:6.1f}%  "
            f"hit={metrics['hit_rate']*100:4.1f}%  n={metrics['n_days']}")
