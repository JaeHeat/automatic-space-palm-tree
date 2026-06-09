"""Strategy 1 - Trend following / momentum (book section 10.4).

The book defines, for assets i with lookback return R_i and volatility sigma_i:

    eta_i = sign(R_i)            (475)   -> optionally tanh-smoothed / market-demeaned
    w_i   = gamma * eta_i/sigma_i (474)  -> inverse-vol sizing
    sum_i |w_i| = 1              (476)   -> gross normalization
    dollar-neutral demean:       (477)   w_i = gamma*(eta_i/sigma_i - mean_j eta_j/sigma_j)

Futures aren't reachable in this environment, so we apply the *same construction*
to a cross-section of major cryptocurrencies (Coinbase daily data, 2022-2026).

Rigor: the momentum lookback T is chosen on an in-sample TRAIN window (2022-2023)
by Sharpe, then the strategy is evaluated out-of-sample (2024-06-09 onward) with
transaction costs. We report long-short (dollar-neutral) and long-only variants
against buy&hold BTC and an equal-weight basket.
"""
import numpy as np
import pandas as pd

from backtest_engine import load_closes, perf_metrics, apply_costs, fmt, ANN

COST_BPS = 10.0          # 10 bps per unit traded notional
REBAL = 5                # rebalance every 5 trading days
VOL_WIN = 30             # trailing window for sigma_i
LOOKBACKS = [20, 30, 60, 90, 120]
TRAIN_END = "2023-12-31"
OOS_START = "2024-01-01"


def build_weights(closes, lookback, vol_win=VOL_WIN, rebal=REBAL,
                  long_short=True, smooth=True, market_demean=True):
    rets = closes.pct_change()
    sigma = rets.rolling(vol_win).std()
    mom = closes.pct_change(lookback)            # R_i over lookback T

    raw = mom.copy()
    if market_demean:                            # eta vs market (eq. after 477)
        raw = raw.sub(raw.mean(axis=1), axis=0)
    if smooth:                                   # tanh-smoothed eta (stability fix)
        kappa = raw.std(axis=1).replace(0, np.nan)
        eta = np.tanh(raw.div(kappa, axis=0))
    else:
        eta = np.sign(raw)

    score = eta / sigma                          # eta_i / sigma_i
    score = score.where(sigma.notna() & mom.notna())

    if long_short:                               # dollar-neutral demean (477)
        score = score.sub(score.mean(axis=1), axis=0)
    else:
        score = score.clip(lower=0)              # long-only

    gross = score.abs().sum(axis=1).replace(0, np.nan)
    w = score.div(gross, axis=0)                 # sum|w| = 1  (476)

    # rebalance only every `rebal` days (hold in between)
    mask = pd.Series(False, index=w.index)
    mask.iloc[::rebal] = True
    w = w.where(mask).ffill()
    return w.fillna(0.0)


def run(closes, lookback, **kw):
    rets = closes.pct_change()
    w = build_weights(closes, lookback, **kw)
    net, turnover = apply_costs(w, rets, COST_BPS)
    return net, turnover, w


def main():
    closes = load_closes()
    rets = closes.pct_change()

    # ---- walk-forward lookback selection on TRAIN ----
    train = closes.loc[:TRAIN_END]
    best_lb, best_sharpe = None, -np.inf
    for lb in LOOKBACKS:
        net, _, _ = run(train, lb, long_short=True)
        s = perf_metrics(net)["sharpe"]
        if np.isfinite(s) and s > best_sharpe:
            best_sharpe, best_lb = s, lb
    print(f"[train 2022-2023] selected lookback T = {best_lb} days "
          f"(train Sharpe {best_sharpe:.2f})\n")

    # ---- out-of-sample evaluation with selected T ----
    results = {}
    for name, kw in [("Long-short (dollar-neutral)", dict(long_short=True)),
                     ("Long-only", dict(long_short=False))]:
        net, turn, _ = run(closes, best_lb, **kw)
        oos = net.loc[OOS_START:]
        m = perf_metrics(oos)
        results[name] = m
        print(f"{name:28s} | OOS {OOS_START}+  {fmt(m)}  "
              f"avg turnover/day={turn.loc[OOS_START:].mean():.3f}")

    # ---- benchmarks (OOS) ----
    btc = perf_metrics(rets["BTC"].loc[OOS_START:])
    eqw = perf_metrics(rets.mean(axis=1).loc[OOS_START:])
    results["Buy&hold BTC"] = btc
    results["Equal-weight basket"] = eqw
    print(f"{'Buy&hold BTC':28s} | OOS {OOS_START}+  {fmt(btc)}")
    print(f"{'Equal-weight basket':28s} | OOS {OOS_START}+  {fmt(eqw)}")
    return best_lb, results


if __name__ == "__main__":
    main()
