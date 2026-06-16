"""
EMA Crossover Backtest — Advanced
=================================

Extends ema_crossover_backtest.py with four robustness studies the first
pass called for:

  1. SHORT SIDE      long/flat  vs  long/short  for every pair.
  2. NEUTRAL UNIVERSE 11 SPDR sector ETFs (XLK..XLRE) — a survivorship-free
                      cross-section of the whole US market, instead of a
                      hand-picked set of known winners.
  3. ENTRY-TIMING SWEEP  execution lag 0 (look-ahead) -> 5 bars, to see how
                      much of any "edge" is just optimistic fills.
  4. REGIME FILTER   trade the crossover only when realized volatility is
                      elevated; otherwise stay long. Tests the earlier
                      finding that crossovers earn their keep in turbulence.

Conventions: signal on daily close; position = signal.shift(lag); returns
close-to-close; 0.05% cost per side. Same metric set as the base script.
"""

import json
import os
import urllib.request
import datetime as dt

import numpy as np
import pandas as pd

PAIRS = [(5, 20), (5, 30), (10, 30), (10, 50), (10, 100), (10, 200), (20, 200)]

# Neutral, survivorship-free universe: the original 9 SPDR sectors (1998),
# plus the two later spin-offs. Equal cross-section of the whole market.
SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
# High-volatility names kept for the regime study.
HIVOL = ["NVDA", "TSLA", "BTC-USD"]

COST_PER_SIDE = 0.0005
TRADING_DAYS = 252
START = "2010-01-01"
CACHE = "/tmp/ema_data_cache"


# ----------------------------------------------------------------------------
# Data (cached to disk so repeated runs don't re-hit Yahoo)
# ----------------------------------------------------------------------------
def fetch(ticker: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{ticker}.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    p2 = int(dt.datetime.now().timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1=0&period2={p2}&interval=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = json.loads(urllib.request.urlopen(req, timeout=30).read())
    res = raw["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(res["timestamp"], unit="s").normalize(),
    ).dropna()
    df = df[df.index >= pd.Timestamp(START)]
    df.to_csv(fp)
    return df


def load(tickers):
    data = {}
    for t in tickers:
        try:
            df = fetch(t)
            if len(df) >= 250:
                data[t] = df
            else:
                print(f"!! {t}: only {len(df)} bars, skipped")
        except Exception as e:
            print(f"!! {t} fetch failed: {e}")
    return data


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(daily_ret: pd.Series, position: pd.Series, n_trades: int) -> dict:
    daily_ret = daily_ret.dropna()
    eq = (1 + daily_ret).cumprod()
    years = len(daily_ret) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily_ret.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dn = daily_ret[daily_ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (daily_ret.mean() * TRADING_DAYS) / dn if dn > 0 else np.nan
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    return {
        "CAGR": cagr, "Max DD": dd, "Sharpe": sharpe, "Sortino": sortino,
        "Calmar": calmar, "Exposure": position.reindex(daily_ret.index).abs().mean(),
        "Trades": n_trades,
    }


def emas(close, fast, slow):
    return (close.ewm(span=fast, adjust=False).mean(),
            close.ewm(span=slow, adjust=False).mean())


def run(df, fast, slow, mode="long", lag=1, regime=None):
    """mode: 'long' (0/1) or 'ls' (-1/+1). regime: None or vol series mask."""
    close = df["close"]
    ef, es = emas(close, fast, slow)
    up = ef > es
    if mode == "long":
        raw = up.astype(float)            # 1 / 0
    else:                                  # long/short
        raw = up.astype(float) * 2 - 1     # +1 / -1

    if regime is not None:                 # trade crossover only in high-vol regime, else long
        raw = raw.where(regime, 1.0)

    position = raw.shift(lag).fillna(0)
    ret = close.pct_change()
    turns = position.diff().abs().fillna(position.abs())
    strat = position * ret - turns * COST_PER_SIDE
    n_entries = int((position.diff().abs() > 0).sum())
    m = metrics(strat, position, n_entries)
    m["_equity"] = (1 + strat.fillna(0)).cumprod()
    return m


def buy_hold(df):
    ret = df["close"].pct_change()
    m = metrics(ret, pd.Series(1.0, index=df.index), 1)
    m["_equity"] = (1 + ret.fillna(0)).cumprod()
    return m


def avg(rows, keys=("CAGR", "Max DD", "Sharpe", "Sortino", "Calmar", "Exposure", "Trades")):
    return {k: np.nanmean([r[k] for r in rows]) for k in keys}


def line(name, m):
    return (f"{name:<16}{m['CAGR']*100:7.2f}%{m['Max DD']*100:9.1f}%"
            f"{m['Sharpe']:7.2f}{m['Sortino']:8.2f}{m['Calmar']:7.2f}"
            f"{m['Exposure']*100:7.1f}%{m['Trades']:6.0f}")


HDR = (f"{'Strategy':<16}{'CAGR':>8}{'MaxDD':>9}{'Sharpe':>7}"
       f"{'Sortino':>8}{'Calmar':>7}{'Expos':>7}{'Trd':>6}")


# ----------------------------------------------------------------------------
# Studies
# ----------------------------------------------------------------------------
def section(t):
    print("\n\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def study_short(data):
    section("1) SHORT SIDE — long/flat vs long/short  (avg over neutral sector universe)")
    print(HDR)
    print(line("Buy & Hold", avg([buy_hold(df) for df in data.values()])))
    print("-" * 74)
    for f, s in PAIRS:
        lf = avg([run(df, f, s, "long") for df in data.values()])
        ls = avg([run(df, f, s, "ls") for df in data.values()])
        print(line(f"{f}/{s} long", lf))
        print(line(f"{f}/{s} L/S", ls))


def study_neutral(data):
    section("2) NEUTRAL UNIVERSE — long/flat crossovers vs Buy & Hold (11 SPDR sectors)")
    bh = avg([buy_hold(df) for df in data.values()])
    print(HDR)
    print(line("Buy & Hold", bh))
    print("-" * 74)
    rank = []
    for f, s in PAIRS:
        m = avg([run(df, f, s, "long") for df in data.values()])
        rank.append((f"{f}/{s}", m))
        print(line(f"{f}/{s} EMA", m))
    print("\nHow many of the 11 sectors did each pair beat Buy & Hold on Sharpe?")
    for f, s in PAIRS:
        wins = sum(run(df, f, s, "long")["Sharpe"] > buy_hold(df)["Sharpe"]
                   for df in data.values())
        print(f"  {f}/{s} EMA: {wins}/{len(data)} sectors")


def study_timing(data):
    section("3) ENTRY-TIMING SWEEP — execution lag (10/30 EMA long, neutral universe)")
    print(f"{'Lag (bars)':<16}{'CAGR':>8}{'MaxDD':>9}{'Sharpe':>7}{'Sortino':>8}{'Calmar':>7}")
    labels = {0: "0 (look-ahead)", 1: "1 (next close)", 2: "2", 3: "3", 5: "5"}
    for lag in [0, 1, 2, 3, 5]:
        m = avg([run(df, 10, 30, "long", lag=lag) for df in data.values()])
        print(f"{labels[lag]:<16}{m['CAGR']*100:7.2f}%{m['Max DD']*100:9.1f}%"
              f"{m['Sharpe']:7.2f}{m['Sortino']:8.2f}{m['Calmar']:7.2f}")
    print("\n(Lag 0 uses the signal-day close to BOTH decide and trade — the")
    print(" optimistic, unrealizable fill. Decay from lag 0->1 = the look-ahead premium.)")


def vol_regime(df, win=20, lookback=252):
    rv = df["close"].pct_change().rolling(win).std() * np.sqrt(TRADING_DAYS)
    return rv > rv.rolling(lookback).median()      # True = high-vol regime


def study_regime(data):
    section("4) REGIME FILTER — crossover only in high-vol regime, else stay long")
    print("Universe: 11 sectors + NVDA, TSLA, BTC-USD\n")
    print(f"{'Strategy':<22}{'CAGR':>8}{'MaxDD':>9}{'Sharpe':>7}{'Sortino':>8}{'Calmar':>7}")
    bh, plain, hybrid = [], [], []
    for df in data.values():
        reg = vol_regime(df)
        bh.append(buy_hold(df))
        plain.append(run(df, 10, 30, "long"))
        hybrid.append(run(df, 10, 30, "long", regime=reg))
    for nm, rows in [("Buy & Hold", bh), ("10/30 crossover", plain),
                     ("10/30 + vol regime", hybrid)]:
        a = avg(rows)
        print(f"{nm:<22}{a['CAGR']*100:7.2f}%{a['Max DD']*100:9.1f}%"
              f"{a['Sharpe']:7.2f}{a['Sortino']:8.2f}{a['Calmar']:7.2f}")


def charts(data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("charts skipped:", e)
        return

    # Equal-weight portfolio equity: B&H vs 10/30 long vs 10/30 L/S (neutral universe)
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.union(df.index)

    def port(mode, **kw):
        eqs = []
        for df in data.values():
            m = run(df, 10, 30, mode, **kw) if mode != "bh" else buy_hold(df)
            r = m["_equity"].pct_change()
            eqs.append(r.reindex(idx))
        pr = pd.concat(eqs, axis=1).mean(axis=1).fillna(0)
        return (1 + pr).cumprod()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(port("bh"), "k--", lw=2.5, label="Buy & Hold")
    ax.plot(port("long"), lw=1.8, label="10/30 EMA long/flat")
    ax.plot(port("ls"), lw=1.8, label="10/30 EMA long/short")
    ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Equal-weight neutral-universe portfolio (11 SPDR sectors)")
    ax.set_ylabel("Growth of $1 (log)"); ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("/home/user/automatic-space-palm-tree/neutral_universe_portfolio.png", dpi=130)
    print("\nSaved chart: neutral_universe_portfolio.png")


def main():
    print(f"ADVANCED EMA CROSSOVER STUDIES | start={START} | cost={COST_PER_SIDE*100:.2f}%/side")
    sectors = load(SECTORS)
    study_short(sectors)
    study_neutral(sectors)
    study_timing(sectors)
    regime_universe = load(SECTORS + HIVOL)
    study_regime(regime_universe)
    charts(sectors)


if __name__ == "__main__":
    main()
