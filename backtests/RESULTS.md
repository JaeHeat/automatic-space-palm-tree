# Backtest Results

Real Coinbase daily data, `2022-01-01 → 2026-06-09`. Costs = 10 bps on traded
notional. Walk-forward, no look-ahead. Annualization = 365. Reproduce with
`python3 run_all.py`.

> These are **honest, unflattering** results. Two of the three strategies do **not**
> beat buy-and-hold BTC on this sample. That is the expected outcome for naïve daily
> implementations of these ideas, and the book itself warns about overfitting.

---

## 1. Trend following / momentum (§10.4) — crypto basket

Lookback `T` selected on TRAIN (2022–2023) by Sharpe → **T = 20 days**.
Out-of-sample = `2024-01-01 →` (891 days).

| Variant | Total ret | CAGR | Vol | Sharpe | Sortino | Max DD | Turnover/day |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Long-short (dollar-neutral)** | +6.1% | +2.5% | 22.1% | **0.22** | 0.27 | −26.1% | 0.135 |
| Long-only | −12.7% | −5.4% | 72.0% | 0.28 | 0.43 | −71.5% | 0.143 |
| *Buy & hold BTC* | +45.6% | +16.6% | 48.6% | 0.56 | 0.85 | −51.2% | — |
| *Equal-weight basket* | −22.6% | −10.0% | 68.3% | 0.19 | 0.28 | −70.9% | — |

**Takeaway.** The dollar-neutral book delivers what it's designed to — **much lower
vol (22% vs 49%) and a shallower drawdown** than directional crypto — but its raw
Sharpe (0.22) trailed simply holding BTC through a strong 2024–25. Cross-sectional
momentum among large-cap coins was weak in this window; the value here is
**diversification / risk control**, not outright return.

## 2. Cryptocurrency ANN quantile classifier (§18.2) — BTC

Rolling 365-day train, refit every 30 days, `K=5` return quantiles, MLP
(ReLU 16×8 → softmax, cross-entropy). Long top quantile / short bottom.

| | Total ret | CAGR | Vol | Sharpe | Max DD | Dir. accuracy (active days) |
|---|---:|---:|---:|---:|---:|---:|
| **ANN long/short** | −34.7% | −9.4% | 26.9% | **−0.23** | −50.5% | **49.7%** |
| *Buy & hold BTC* | +45.8% | +9.1% | 51.4% | 0.43 | −66.8% | — |

**Takeaway.** Directional accuracy is **49.7% — indistinguishable from a coin flip.**
With no predictive edge, the ~0.34/day turnover bleeds money through costs. Daily
next-day BTC direction from technical features alone is, on this sample, **noise.**
(The book applies this to 15-min bars with far more data; daily bars give the net
far too few observations to learn from.)

## 3. Sentiment naïve Bayes (§18.3) — controlled simulation

**Not a live-data backtest** — no historical tweet archive is reachable here. This
sweeps a synthetic tweet stream with a tunable `signal` = fraction of days whose
tweets carry true next-day information, trading the **real** BTC series.

| `signal` | Dir. accuracy | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| 0.00 (pure noise) | 49.9% | −18.9% | −0.29 | −80.1% |
| 0.15 | 56.1% | +44.4% | 1.09 | −29.9% |
| 0.35 | 65.6% | +167.9% | 2.61 | −36.6% |
| 0.60 | 81.5% | +1159.5% | 6.73 | −18.2% |

**Takeaway.** This is a **correctness test, not a P&L claim.** At `signal=0` the
strategy is flat-to-negative after costs (no spurious edge); as genuine information
is injected, accuracy and Sharpe rise monotonically — confirming the Bernoulli NB
math (eqs. 543–546) and the trading wrapper are implemented correctly. Whether real
Twitter sentiment supplies any of that `signal` is an **empirical question that needs
a real tweet dataset** to answer.

---

## Honest conclusions

1. **Risk control, not alpha.** The only strategy that behaved well on a risk-adjusted
   basis was dollar-neutral momentum — and its edge was *lower volatility*, not higher
   return, in a period where BTC simply went up.
2. **Daily ML on price alone = noise.** The ANN had literally coin-flip accuracy. More
   data (intraday), richer features, or an ensemble would be needed before this is
   worth anything.
3. **Sentiment is unproven here.** The pipeline is correct and ready; it needs a real
   historical tweet/sentiment feed (an API key or dataset) to test for an actual edge.

### Natural next steps
- Intraday bars (15-min, as in the book) for the ANN, plus walk-forward hyperparameter
  search and an ensemble.
- A real crypto news/social-sentiment feed for §18.3.
- Add funding-rate / carry signals and risk-parity sizing to the momentum book.
- Volatility-target overlay and a regime filter (e.g., only long-only when BTC > 200d MA).
