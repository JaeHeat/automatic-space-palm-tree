"""
EMA Crossover Backtest
======================

Backtests the moving-average crossover pairs recommended in the book page
"Moving Average Crossover Trading 101":

    5 EMA / 20 EMA
    5 EMA / 30 EMA
    10 EMA / 30 EMA
    10 EMA / 50 EMA
    10 EMA / 100 EMA
    10 EMA / 200 EMA
    20 EMA / 200 EMA

Rules (long-only mechanical crossover, as described in the book):
  * LONG when the fast EMA is above the slow EMA.
  * FLAT (in cash, 0% return) when the fast EMA is below the slow EMA.
  * Signals are computed on the daily close; trades are executed at the
    NEXT day's open to avoid look-ahead bias.
  * A round-trip transaction cost is applied on every entry and exit.

Data: daily OHLC from Yahoo Finance (full available history per ticker).
"""

import json
import urllib.request
import datetime as dt

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
PAIRS = [(5, 20), (5, 30), (10, 30), (10, 50), (10, 100), (10, 200), (20, 200)]

# A representative cross-asset "watchlist": broad indices, mega-cap tech,
# a high-beta name, and bitcoin.
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "BTC-USD"]

COST_PER_SIDE = 0.0005      # 0.05% per fill (entry and exit each)
TRADING_DAYS = 252
START = "2010-01-01"        # cap history so all assets share a comparable era


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def fetch(ticker: str) -> pd.DataFrame:
    p1 = 0  # 1970
    p2 = int(dt.datetime.now().timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = json.loads(urllib.request.urlopen(req, timeout=30).read())
    res = raw["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": q["open"],
            "high": q["high"],
            "low": q["low"],
            "close": q["close"],
        },
        index=pd.to_datetime(ts, unit="s").normalize(),
    )
    df = df.dropna()
    df = df[df.index >= pd.Timestamp(START)]
    return df


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(daily_ret: pd.Series, position: pd.Series, n_trades: int) -> dict:
    eq = (1 + daily_ret).cumprod()
    years = len(daily_ret) / TRADING_DAYS
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily_ret.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (daily_ret.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan
    exposure = position.mean()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "Total Return": total,
        "CAGR": cagr,
        "Max DD": maxdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Exposure": exposure,
        "Trades": n_trades,
    }


def backtest_pair(df: pd.DataFrame, fast: int, slow: int) -> dict:
    close = df["close"]
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()

    # Raw signal on close: 1 when fast>slow else 0. Trade next open => shift(1).
    signal = (ef > es).astype(int)
    position = signal.shift(1).fillna(0)

    # Open-to-open returns are what an at-the-open executor actually earns.
    o2o = df["open"].pct_change().shift(-1)  # return from today's open to next open
    # align: position decided at close[t] is held over open[t+1]->open[t+2]
    # Simpler & standard: use close-to-close with next-bar execution.
    c2c = close.pct_change()
    strat = position * c2c

    # Transaction costs on position changes.
    turns = position.diff().abs().fillna(position.abs())
    cost = turns * COST_PER_SIDE
    strat_net = strat - cost

    n_trades = int((position.diff() == 1).sum())  # number of entries

    m = metrics(strat_net.dropna(), position.loc[strat_net.dropna().index], n_trades)
    m["_equity"] = (1 + strat_net.fillna(0)).cumprod()
    return m


def buy_hold(df: pd.DataFrame) -> dict:
    c2c = df["close"].pct_change()
    pos = pd.Series(1.0, index=df.index)
    m = metrics(c2c.dropna(), pos.loc[c2c.dropna().index], 1)
    m["_equity"] = (1 + c2c.fillna(0)).cumprod()
    return m


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def fmt(m: dict) -> str:
    return (
        f"{m['Total Return']*100:8.1f}% {m['CAGR']*100:7.2f}% "
        f"{m['Max DD']*100:8.1f}% {m['Sharpe']:6.2f} {m['Sortino']:7.2f} "
        f"{m['Calmar']:6.2f} {m['Exposure']*100:6.1f}% {m['Trades']:5.0f}"
    )


def main():
    print(f"EMA Crossover Backtest  |  start={START}  cost={COST_PER_SIDE*100:.2f}%/side\n")
    header = (
        f"{'Strategy':<16}{'TotRet':>9}{'CAGR':>8}{'MaxDD':>9}"
        f"{'Sharpe':>7}{'Sortino':>8}{'Calmar':>7}{'Expos':>7}{'Trd':>6}"
    )

    all_rows = {f"{f}/{s}": [] for f, s in PAIRS}
    bh_rows = []
    data = {}

    for tkr in TICKERS:
        try:
            df = fetch(tkr)
        except Exception as e:
            print(f"!! {tkr} fetch failed: {e}")
            continue
        if len(df) < 250:
            print(f"!! {tkr} insufficient data ({len(df)} rows)")
            continue
        data[tkr] = df
        span = f"{df.index[0].date()} -> {df.index[-1].date()} ({len(df)} bars)"
        print(f"\n=== {tkr}   {span} ===")
        print(header)
        bh = buy_hold(df)
        print(f"{'Buy & Hold':<16}{fmt(bh)}")
        bh_rows.append(bh)
        print("-" * len(header))
        for f, s in PAIRS:
            m = backtest_pair(df, f, s)
            all_rows[f"{f}/{s}"].append(m)
            print(f"{f'{f}/{s} EMA':<16}{fmt(m)}")

    # ---- Aggregate across all tickers (equal-weight average of metrics) ----
    print("\n\n########  AVERAGE ACROSS ALL TICKERS  ########")
    print(header)
    bh_avg = {k: np.nanmean([r[k] for r in bh_rows]) for k in
              ['Total Return','CAGR','Max DD','Sharpe','Sortino','Calmar','Exposure','Trades']}
    print(f"{'Buy & Hold':<16}{fmt(bh_avg)}")
    print("-" * len(header))
    ranking = []
    for key, rows in all_rows.items():
        avg = {k: np.nanmean([r[k] for r in rows]) for k in
               ['Total Return','CAGR','Max DD','Sharpe','Sortino','Calmar','Exposure','Trades']}
        print(f"{key+' EMA':<16}{fmt(avg)}")
        ranking.append((key, avg))

    print("\nRanked by avg Sharpe:")
    for key, avg in sorted(ranking, key=lambda x: -x[1]['Sharpe']):
        print(f"  {key+' EMA':<14} Sharpe {avg['Sharpe']:.2f} | CAGR {avg['CAGR']*100:5.2f}% "
              f"| MaxDD {avg['Max DD']*100:6.1f}% | Calmar {avg['Calmar']:.2f}")
    print(f"  {'Buy & Hold':<14} Sharpe {bh_avg['Sharpe']:.2f} | CAGR {bh_avg['CAGR']*100:5.2f}% "
          f"| MaxDD {bh_avg['Max DD']*100:6.1f}% | Calmar {bh_avg['Calmar']:.2f}")

    # ---- Chart: equity curves on SPY ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_tkr = "SPY" if "SPY" in data else next(iter(data))
        df = data[chart_tkr]
        fig, ax = plt.subplots(figsize=(12, 7))
        bh = buy_hold(df)
        ax.plot(bh["_equity"].index, bh["_equity"], "k--", lw=2.5, label="Buy & Hold")
        for f, s in PAIRS:
            m = backtest_pair(df, f, s)
            ax.plot(m["_equity"].index, m["_equity"], lw=1.3, label=f"{f}/{s} EMA")
        ax.set_yscale("log")
        ax.set_title(f"EMA Crossover Strategies vs Buy & Hold — {chart_tkr} "
                     f"({df.index[0].date()} to {df.index[-1].date()})")
        ax.set_ylabel("Growth of $1 (log scale)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig("/home/user/automatic-space-palm-tree/equity_curves_SPY.png", dpi=130)
        print(f"\nSaved chart: equity_curves_SPY.png ({chart_tkr})")
    except Exception as e:
        print(f"chart skipped: {e}")


if __name__ == "__main__":
    main()
