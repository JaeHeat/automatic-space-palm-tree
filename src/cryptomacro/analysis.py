"""The four macro/crypto analyses, returning plain data structures.

Each ``analyze_*`` function fetches what it needs, computes results, and returns
a dataclass holding tidy frames plus a human-readable ``summary`` string. Scripts
and notebooks call these; plotting is kept separate in :mod:`cryptomacro.plotting`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import data, transforms as tf


# ---------------------------------------------------------------------------
# 1. Cycle comparison (current vs previous bears/bulls)
# ---------------------------------------------------------------------------

@dataclass
class CycleComparison:
    aligned_price: pd.DataFrame      # rebased=100 at each halving, indexed by day offset
    drawdowns: pd.DataFrame          # drawdown-from-ATH per cycle, indexed by day offset
    stats: pd.DataFrame              # per-cycle max drawdown, peak gain, etc.
    summary: str


def analyze_cycles(window_days: int = 1300) -> CycleComparison:
    btc = data.get_btc()
    # Use halvings we actually have price history for (CBBTCUSD starts 2014-12).
    anchors = {k: v for k, v in data.HALVING_DATES.items() if v >= btc.index.min()}

    aligned = tf.align_to_anchor(btc, anchors, window_days=window_days, rebase_at_anchor=True)

    # Drawdown profile per cycle, computed within each post-halving slice.
    dd_cols = {}
    for label, anchor in anchors.items():
        seg = btc.loc[anchor: anchor + pd.Timedelta(days=window_days)]
        if seg.empty:
            continue
        dd = tf.drawdown(seg)
        dd.index = (dd.index - anchor).days
        dd_cols[label] = dd[~dd.index.duplicated(keep="first")]
    drawdowns = pd.DataFrame(dd_cols).sort_index()

    rows = []
    for label in aligned.columns:
        col = aligned[label].dropna()
        ddcol = drawdowns[label].dropna() if label in drawdowns else pd.Series(dtype=float)
        rows.append({
            "cycle": label,
            "days_tracked": int(col.index.max()) if not col.empty else 0,
            "peak_gain_x": round(col.max() / 100.0, 2) if not col.empty else np.nan,
            "max_drawdown_pct": round(ddcol.min(), 1) if not ddcol.empty else np.nan,
        })
    stats = pd.DataFrame(rows).set_index("cycle")

    cur = aligned.columns[-1]
    summary = (
        f"Aligned {len(aligned.columns)} post-halving cycles on a common axis.\n"
        f"Per-cycle peak gain and max drawdown:\n{stats.to_string()}\n"
        f"Current cycle ({cur}) tracked {stats.loc[cur, 'days_tracked']} days "
        f"since its halving."
    )
    return CycleComparison(aligned, drawdowns, stats, summary)


# ---------------------------------------------------------------------------
# 2. BTC vs Global M2 liquidity (bull vs bear regimes + lead-lag)
# ---------------------------------------------------------------------------

@dataclass
class M2Liquidity:
    merged: pd.DataFrame             # monthly BTC + Global M2 (USD trillions)
    rolling_corr: pd.Series          # rolling corr of YoY growth rates
    lead_lag: pd.DataFrame           # cross-correlation table (M2 leading BTC)
    best_lag_months: int
    best_lag_corr: float
    regime_corr: pd.Series           # corr of monthly returns within bull vs bear
    summary: str


def analyze_m2_liquidity(roll_window: int = 12, max_lag_months: int = 18) -> M2Liquidity:
    btc = data.get_btc()
    basket = data.DEFAULT_M2_BASKET
    m2 = data.get_global_m2(basket)["Global"]
    basket_label = " + ".join(c.name for c in basket)

    btc_m = btc.resample("ME").last()
    merged = pd.concat([btc_m.rename("BTC"), m2.rename("GlobalM2")], axis=1).dropna()

    # Correlate *growth rates* (YoY %) to avoid spurious trend-on-trend correlation.
    btc_yoy = merged["BTC"].pct_change(12) * 100
    m2_yoy = merged["GlobalM2"].pct_change(12) * 100
    rolling_corr = tf.rolling_correlation(m2_yoy, btc_yoy, window=roll_window)

    # Lead-lag on YoY growth: does M2 growth lead BTC growth?
    lead_lag = tf.lead_lag_correlation(m2_yoy.dropna(), btc_yoy.dropna(), max_lag=max_lag_months)
    best_lag, best_corr = tf.best_lead_lag(m2_yoy.dropna(), btc_yoy.dropna(), max_lag=max_lag_months)

    # Regime-conditional correlation of monthly *returns* (bull vs bear by 200d MA).
    regime_d = tf.regime_by_ma(btc, window=200).resample("ME").last()
    btc_ret = merged["BTC"].pct_change()
    m2_ret = merged["GlobalM2"].pct_change()
    rdf = pd.concat([btc_ret.rename("btc"), m2_ret.rename("m2"), regime_d.rename("regime")], axis=1).dropna()
    regime_corr = pd.Series(
        {reg: g["btc"].corr(g["m2"]) for reg, g in rdf.groupby("regime")},
        name="btc_m2_return_corr",
    )

    summary = (
        f"Global M2 basket: {basket_label} (broad money, converted to USD).\n"
        f"Window: {merged.index.min().date()} .. {merged.index.max().date()} "
        f"({len(merged)} months).\n"
        f"Latest Global M2: {merged['GlobalM2'].iloc[-1]:.1f}T USD; "
        f"BTC: ${merged['BTC'].iloc[-1]:,.0f}.\n"
        f"Peak lead-lag: Global-M2 YoY growth leads BTC YoY growth by "
        f"{best_lag} months (corr={best_corr:.2f}).\n"
        f"Return correlation by regime:\n{regime_corr.to_string()}"
    )
    return M2Liquidity(merged, rolling_corr, lead_lag, best_lag, best_corr, regime_corr, summary)


# ---------------------------------------------------------------------------
# 3. BTC vs Nasdaq-100 (rolling correlation + OLS beta)
# ---------------------------------------------------------------------------

@dataclass
class NQCorrelation:
    rolling_corr: pd.Series          # rolling corr of daily returns
    rolling_beta: pd.Series          # rolling OLS beta of BTC on NDX
    full_ols: dict                   # alpha, beta, r2, t-stats over full sample
    summary: str


def _ols(y: pd.Series, x: pd.Series) -> dict:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    model = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
    return {
        "alpha": model.params["const"],
        "beta": model.params["x"],
        "alpha_t": model.tvalues["const"],
        "beta_t": model.tvalues["x"],
        "r2": model.rsquared,
        "n": int(model.nobs),
    }


def analyze_nq(roll_window: int = 90) -> NQCorrelation:
    btc = data.get_btc()
    ndx = data.get_nasdaq100()

    btc_r = tf.returns(btc)
    ndx_r = tf.returns(ndx)
    df = pd.concat([btc_r.rename("btc"), ndx_r.rename("ndx")], axis=1).dropna()

    rolling_corr = df["btc"].rolling(roll_window).corr(df["ndx"]).dropna()

    # Rolling beta = rolling cov / rolling var.
    cov = df["btc"].rolling(roll_window).cov(df["ndx"])
    var = df["ndx"].rolling(roll_window).var()
    rolling_beta = (cov / var).dropna()
    rolling_beta.name = "beta"

    full_ols = _ols(df["btc"], df["ndx"])

    summary = (
        f"BTC vs Nasdaq-100, daily returns, {df.index.min().date()}..{df.index.max().date()}.\n"
        f"Full-sample OLS: BTC_ret = {full_ols['alpha']*100:.3f}%/day "
        f"+ {full_ols['beta']:.2f}*NDX_ret  (R^2={full_ols['r2']:.2f}, "
        f"beta t={full_ols['beta_t']:.1f}, n={full_ols['n']}).\n"
        f"Rolling {roll_window}d correlation latest={rolling_corr.iloc[-1]:.2f}, "
        f"range [{rolling_corr.min():.2f}, {rolling_corr.max():.2f}]."
    )
    return NQCorrelation(rolling_corr, rolling_beta, full_ols, summary)


# ---------------------------------------------------------------------------
# 4. Business cycle: BTC returns conditioned on macro regime
# ---------------------------------------------------------------------------

@dataclass
class BusinessCycle:
    indicators: pd.DataFrame         # monthly macro indicators + BTC
    by_yield_curve: pd.DataFrame     # BTC monthly return stats when curve inverted vs not
    by_growth: pd.DataFrame          # BTC stats when industrial production accelerating vs not
    by_recession: pd.DataFrame       # BTC stats in NBER recession vs expansion
    summary: str


def _return_stats(monthly_ret: pd.Series, mask: pd.Series, labels: tuple[str, str]) -> pd.DataFrame:
    aligned = pd.concat([monthly_ret.rename("ret"), mask.rename("flag")], axis=1).dropna()
    out = {}
    for flag_val, name in zip([True, False], labels):
        grp = aligned.loc[aligned["flag"] == flag_val, "ret"]
        out[name] = {
            "months": int(grp.shape[0]),
            "mean_monthly_%": round(grp.mean() * 100, 2),
            "median_%": round(grp.median() * 100, 2),
            "annualized_%": round(((1 + grp.mean()) ** 12 - 1) * 100, 1) if grp.shape[0] else np.nan,
            "vol_monthly_%": round(grp.std() * 100, 2),
            "hit_rate_%": round((grp > 0).mean() * 100, 1) if grp.shape[0] else np.nan,
        }
    return pd.DataFrame(out).T


def analyze_business_cycle() -> BusinessCycle:
    btc = data.get_btc()
    bc = data.get_business_cycle()

    btc_m = btc.resample("ME").last()
    ret = btc_m.pct_change()
    ind = bc.join(btc_m.rename("BTC")).join(ret.rename("BTC_ret"))

    curve_inverted = bc["yield_curve_10y_2y"] < 0
    growth_accel = bc["industrial_production_yoy"].diff() > 0
    in_recession = bc["recession"] > 0.5

    by_curve = _return_stats(ret, curve_inverted, ("yield_curve_inverted", "curve_normal"))
    by_growth = _return_stats(ret, growth_accel, ("IP_growth_accelerating", "IP_growth_slowing"))
    by_rec = _return_stats(ret, in_recession, ("NBER_recession", "expansion"))

    summary = (
        "BTC monthly returns conditioned on the US business cycle:\n"
        f"-- Yield curve --\n{by_curve.to_string()}\n"
        f"-- Industrial-production momentum --\n{by_growth.to_string()}\n"
        f"-- NBER recession flag --\n{by_rec.to_string()}"
    )
    return BusinessCycle(ind, by_curve, by_growth, by_rec, summary)
