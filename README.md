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
