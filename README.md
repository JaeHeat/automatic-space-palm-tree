# cryptomacro — macro/crypto data analysis

A small, reusable toolkit for the kind of macro questions crypto analysts actually
ask: *Where is BTC in its cycle vs previous bears? How tied is it to global
liquidity (M2)? Does liquidity lead price? How correlated is it to the Nasdaq?
How does it behave across the business cycle?*

It grew out of a simpler question — *"can linear regression be used for crypto?"* —
and answers it concretely: OLS shows up here as **BTC's beta to the Nasdaq-100**
and as **regime-conditional regressions**, while the more durable signal comes
from correlation, lead-lag and cycle-alignment analysis.

Everything runs off a **single, keyless data source — FRED's CSV endpoint** — so
there are no API keys to manage. BTC and the Nasdaq-100 are pulled from FRED too
(`CBBTCUSD`, `NASDAQ100`), alongside M2, FX and business-cycle series.

## The four analyses

| # | Module entry point | What it does | Method |
|---|--------------------|--------------|--------|
| 1 | `analysis.analyze_cycles()` | Overlay current cycle vs previous bull/bear markets, aligned to each halving; compare drawdown profiles | rebasing, drawdown, cycle alignment |
| 2 | `analysis.analyze_m2_liquidity()` | BTC vs **Global M2** (US+EU+CN+JP in USD); rolling correlation, does liquidity *lead* price, bull vs bear | YoY-growth correlation, lead-lag cross-correlation, regime split |
| 3 | `analysis.analyze_nq()` | BTC vs **Nasdaq-100**; how much "leveraged tech risk" BTC carries and how it drifts | rolling correlation, rolling + full-sample **OLS beta** |
| 4 | `analysis.analyze_business_cycle()` | BTC monthly returns conditioned on yield curve, industrial-production momentum, NBER recessions | regime-conditional return stats |

## Quick start

```bash
pip install -r requirements.txt

# (optional but recommended) warm the local data cache one series at a time.
# Helpful on networks that rate-limit bursts of requests.
python scripts/prefetch.py

# Run any analysis — prints a summary and saves a chart to outputs/
python scripts/run_cycle_comparison.py
python scripts/run_m2_liquidity.py
python scripts/run_nq_correlation.py
python scripts/run_business_cycle.py
```

Or explore interactively:

```bash
jupyter lab notebooks/
```

## Using the library directly

```python
import sys; sys.path.insert(0, "src")
from cryptomacro import analysis, plotting

res = analysis.analyze_nq()
print(res.summary)          # alpha, beta, R², rolling-corr range
print(res.full_ols)         # {'alpha':..., 'beta':..., 'r2':...}
fig = plotting.plot_nq_correlation(res)
fig.savefig("nq.png")
```

## Project layout

```
src/cryptomacro/
  fred.py         keyless FRED CSV fetcher (cache + retry/backoff + curl fallback)
  data.py         BTC, Nasdaq-100, Global M2 basket, business-cycle indicators
  transforms.py   returns, drawdown, rolling/lead-lag correlation, regimes, cycle alignment
  analysis.py     the four analyses -> dataclasses with tidy frames + a text summary
  plotting.py     matplotlib charts (one per analysis)
scripts/          runnable entry points (+ prefetch.py to warm the cache)
notebooks/        one notebook per analysis, calling the shared modules
```

## Data sources (all via FRED, no API key)

| Series | FRED id | Notes |
|--------|---------|-------|
| BTC/USD | `CBBTCUSD` | Coinbase; daily from 2014-12 |
| Nasdaq-100 | `NASDAQ100` | proxy for NQ futures |
| US M2 | `M2SL` | billions USD |
| Euro / China / Japan M2 | `MYAGM2EZM196N`, `MYAGM2CNM189N`, `MYAGM2JPM189S` | converted to USD via FX |
| FX | `DEXUSEU`, `DEXCHUS`, `DEXJPUS` | for the global-M2 conversion |
| Business cycle | `T10Y2Y`, `INDPRO`, `UNRATE`, `NFCI`, `USREC` | yield curve, output, recession flag |

## Caveats (read these before trading on it)

- **Correlation ≠ prediction.** Crypto relationships are reflexive and decay; treat
  these as descriptive, not a crystal ball. Validate any predictive use out-of-sample.
- **Stationarity.** We deliberately correlate *returns* and *YoY growth rates*, not
  raw price/M2 levels, to avoid spurious trend-on-trend correlation.
- **History is short.** BTC daily on FRED starts 2014-12, so cycle comparison covers
  the 2016/2020/2024 halvings (the 2012 cycle and 2013 top aren't in the data).
- **Global M2 is an approximation.** It's a 4-bloc basket converted at spot FX; the
  popular "global liquidity" charts vary in composition and FX treatment.
- **Lead-lag is in-sample.** The "M2 leads BTC by N months" figure is fit on the whole
  history; it is suggestive, not a guarantee it persists.
