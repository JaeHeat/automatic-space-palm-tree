# Backtests — three strategies from *151 Trading Strategies*

Runnable, cost-aware, walk-forward backtests for three strategies described in
`../trend-following-and-crypto-strategies.md`:

| # | Strategy | Book § | Script |
|---|----------|--------|--------|
| 1 | Trend following / momentum | 10.4 | `strat_trend_following.py` |
| 2 | Cryptocurrency ANN quantile classifier | 18.2 | `strat_crypto_ann.py` |
| 3 | Sentiment analysis, Bernoulli naïve Bayes | 18.3 | `strat_sentiment_nb.py` |

## Quick start

```bash
pip install -r requirements.txt
python3 fetch_data.py     # refresh data/ from Coinbase (optional; CSVs are committed)
python3 run_all.py        # run all three and print the consolidated report
```

## Data

Real **daily OHLCV from the Coinbase Exchange public API** (no key, no VPN),
`2022-01-01 → 2026-06-09`, cached in `data/`:

`BTC ETH SOL XRP ADA DOGE LTC BCH LINK AVAX DOT XLM` (all but XRP have full history;
XRP starts 2023-07 due to its Coinbase relisting and is handled per-day).

### Honest data caveats

- **No futures data.** Yahoo / Stooq / Binance are blocked from this environment, so
  the §10.4 *futures* trend-following construction is applied to a **crypto basket**
  instead — the formula is identical, only the universe differs.
- **No historical tweet archive.** The §18.3 sentiment strategy therefore runs as a
  **controlled simulation** on the real BTC price series with a synthetic, tunable
  tweet stream. It validates that the classifier is implemented correctly; it is
  **not** evidence of a live-Twitter edge. See the script header for details.

## Method (rigor)

- **Walk-forward, no look-ahead.** Models/quantile thresholds/scalers are fit on a
  trailing TRAIN window and applied forward; positions set at close `t` earn the
  `t → t+1` return (positions are shifted before multiplying by returns).
- **Transaction costs:** 10 bps charged on traded notional (`Σ|Δw|`) every day.
- **Metrics:** total return, CAGR, annualized vol, Sharpe, Sortino, max drawdown,
  Calmar, hit rate — all in `backtest_engine.py`, annualized at 365.
- **Benchmarks:** buy-and-hold BTC and an equal-weight basket.

See `RESULTS.md` for the latest numbers and interpretation.
