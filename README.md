# qstack

An **open-source quant stack** — a drop-in alternative to the four commercial
Quant Science libraries (QSConnect, QSResearch, QSWorkflow, Omega), built
entirely on open tooling.

| Quant Science | qstack module     | Does                                            | Open backends |
|---------------|-------------------|-------------------------------------------------|---------------|
| QSConnect     | `qstack.connect`  | Build/query a quant research database           | SQLite (default) or DuckDB; data via synthetic / yfinance / ccxt |
| QSResearch    | `qstack.research` | Research & backtest strategies (incl. ML-ready) | pure-numpy backtester, optional vectorbt |
| QSWorkflow    | `qstack.workflow` | Automate the end-to-end process                 | `Pipeline` orchestrator |
| Omega         | `qstack.omega`    | Execute trades with Python                       | paper broker (default), ccxt, alpaca-py |

> The original four are paid products sold via quantscience.io. This repo
> reimplements the *capabilities* on free, permissively licensed libraries — no
> license key required.

## Why it runs out of the box

The core package depends only on `numpy` and `pandas`. With **zero** extra
dependencies you get a deterministic synthetic data source, a SQLite-backed
store, a vectorized backtester, and an in-process paper broker — so a full
ingest → research → execute run needs no network, no API key, and no credentials.
Install an extra only when you want to point a stage at a live backend.

## Install

```bash
pip install -e .                 # core (numpy + pandas)
pip install -e '.[data]'         # + yfinance (equities) + duckdb (columnar store)
pip install -e '.[crypto]'       # + ccxt (crypto data & execution)
pip install -e '.[backtest]'     # + vectorbt engine
pip install -e '.[broker]'       # + alpaca-py (equities execution)
pip install -e '.[all]'          # everything
```

## Quick start

```python
from functools import partial
from qstack import Pipeline, DataStore, PaperBroker, get_source
from qstack.research import sma_crossover

pipeline = Pipeline(
    symbol="BTC-DEMO",
    strategy=partial(sma_crossover, fast=20, slow=50),
    source=get_source("synthetic"),   # -> get_source("ccxt") with the crypto extra
    store=DataStore(":memory:"),       # -> DataStore("research.db") to persist
    broker=PaperBroker(cash=100_000),  # -> CCXTBroker(...) / AlpacaBroker(...) to go live
)

result = pipeline.run(lookback_days=730)
print(result)
print(result.backtest.summary())
```

Or from the terminal:

```bash
qstack run --symbol BTC --strategy sma --fast 20 --slow 50
qstack run --symbol AAPL --source yfinance --lookback 730   # needs the [data] extra
```

Full demo: [`examples/end_to_end.py`](examples/end_to_end.py).

### Backtest real data and compare strategies

[`examples/backtest_demo.py`](examples/backtest_demo.py) ingests real prices,
backtests every strategy against a buy & hold benchmark (net of fees +
slippage), prints a performance table, and saves an equity-curve chart.

```bash
pip install -e '.[data]'    # adds yfinance + matplotlib
python examples/backtest_demo.py --symbol SPY --source yfinance --start 2015-01-01 --end 2024-01-01
python examples/backtest_demo.py --symbol BTC --source synthetic   # offline, no network
```

## The four layers

### `qstack.connect` — research database (QSConnect)
`DataStore` gives a uniform `write` / `read` / `ingest` API over DuckDB or
SQLite. `get_source(...)` returns a data fetcher (`synthetic`, `yfinance`,
`ccxt`) that all emit the same normalised OHLCV frame, so the rest of the stack
is backend-agnostic.

### `qstack.research` — strategies & backtesting (QSResearch)
A strategy is any `(df) -> position series in {-1,0,+1}`. Ships with
`sma_crossover`, `rsi_reversion`, and `momentum`. `Backtest` applies a one-bar
lag (no look-ahead), charges fees + slippage, and reports CAGR, Sharpe, max
drawdown, win rate, and trade count.

### Intraday futures: Opening Range Breakout (ORB)

`qstack.research.orb` is a session-aware intraday day-trading engine sized in
real futures economics (point value, ticks, commission, slippage). It trades the
first breakout of the opening range, with a stop and optional target sized as
multiples of the range, and flats by the session close.

```python
from qstack.connect import load_ohlcv_csv
from qstack.research import backtest_orb, ORBParams, NQ

df = load_ohlcv_csv("nq_5m.csv")                       # Databento 5m export -> ET index
res = backtest_orb(df, ORBParams(or_minutes=30, direction="both",
                                 stop_mult=1.0, target_mult=2.0), NQ)
print(res)                                             # trades, net P&L, PF, win%, maxDD, Sharpe
```

Optional filters keep a raw ORB from bleeding in chop: a **trend filter**
(`trend_ma` — only trade with the daily trend), a **volatility filter**
(`vol_min_frac` — skip small opening ranges), and an **entry cutoff**
(`entry_cutoff` — no new entries late in the day).

[`examples/orb_futures.py`](examples/orb_futures.py) evaluates the parameter
grid two ways so a "profitable" claim isn't curve-fit: a single TRAIN/TEST
split, and a rolling **walk-forward** (`walk_forward()` — re-optimize on a
trailing year, trade the next quarter, repeat) that produces a fully
out-of-sample track record.

```bash
python examples/orb_futures.py  --csv nq_5m.csv --instrument NQ --split 2025-01-01 --plot orb_nq.png
python examples/orb_analysis.py --csv nq_5m.csv   # dissect losses: win-rate/PF by side, time, trend, OR size
```

Extra entry/exit controls for tuning the win-rate / risk-reward tradeoff:
`confirm_close` (require the breakout bar to close beyond the range — filters
false breakouts), `breakeven_at` (move the stop to entry after a favourable
move), and `skip_monday`.

### `qstack.workflow` — orchestration (QSWorkflow)
`Pipeline` chains ingest → research → execute into one repeatable `run()`, or
step through the stages individually.

### `qstack.omega` — execution (Omega)
One `Broker` interface, three implementations: `PaperBroker` (default
simulator), `CCXTBroker` (crypto), and `AlpacaBroker` (equities). Swap the
broker without touching strategy or workflow code.

## Going live

Each layer swaps independently — keep everything else the same:

```python
from qstack.connect import get_source
from qstack.omega import CCXTBroker

source = get_source("ccxt", exchange="binance")          # real crypto data
broker = CCXTBroker("binance", api_key=..., secret=...)   # real execution
```

> ⚠️ Live trading risks real capital. Validate on the paper broker first, and
> keep credentials in environment variables — never commit them.

## Tests

```bash
pip install pytest
pytest -q
```

## License

MIT.
