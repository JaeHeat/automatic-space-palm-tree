"""Matplotlib chart helpers. Each returns the Matplotlib Figure so callers can
save it (scripts) or display it inline (notebooks)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # safe default for headless/script use; notebooks override.
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import BusinessCycle, CycleComparison, M2Liquidity, NQCorrelation  # noqa: E402


def plot_cycle_comparison(res: CycleComparison):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    for col in res.aligned_price.columns:
        ax1.plot(res.aligned_price.index, res.aligned_price[col], label=f"{col} halving")
    ax1.set_yscale("log")
    ax1.set_title("BTC cycles aligned to halving (rebased=100 at halving, log scale)")
    ax1.set_xlabel("days since halving")
    ax1.set_ylabel("price (rebased)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    for col in res.drawdowns.columns:
        ax2.plot(res.drawdowns.index, res.drawdowns[col], label=f"{col} cycle")
    ax2.set_title("Drawdown from all-time high, by cycle")
    ax2.set_xlabel("days since halving")
    ax2.set_ylabel("drawdown %")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_m2_liquidity(res: M2Liquidity):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11))

    ax1b = ax1.twinx()
    ax1.plot(res.merged.index, res.merged["BTC"], color="tab:orange", label="BTC")
    ax1.set_yscale("log")
    ax1.set_ylabel("BTC ($, log)", color="tab:orange")
    ax1b.plot(res.merged.index, res.merged["GlobalM2"], color="tab:blue", label="Global M2")
    ax1b.set_ylabel("Global M2 ($T)", color="tab:blue")
    ax1.set_title("BTC vs Global M2 (US+EU+CN+JP, USD)")

    ax2.plot(res.rolling_corr.index, res.rolling_corr.values, color="tab:purple")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_title("Rolling correlation of YoY growth rates (BTC vs Global M2)")
    ax2.set_ylabel("correlation")
    ax2.grid(True, alpha=0.3)

    ax3.bar(res.lead_lag["lag"], res.lead_lag["corr"], color="tab:green")
    ax3.axvline(res.best_lag_months, color="red", ls="--",
                label=f"peak: {res.best_lag_months} mo (r={res.best_lag_corr:.2f})")
    ax3.set_title("Lead-lag: Global-M2 growth leading BTC growth")
    ax3.set_xlabel("months M2 leads BTC")
    ax3.set_ylabel("correlation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_nq_correlation(res: NQCorrelation):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.plot(res.rolling_corr.index, res.rolling_corr.values, color="tab:blue")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_title("BTC vs Nasdaq-100: rolling correlation of daily returns")
    ax1.set_ylabel("correlation")
    ax1.grid(True, alpha=0.3)

    ax2.plot(res.rolling_beta.index, res.rolling_beta.values, color="tab:red")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axhline(1, color="gray", lw=0.8, ls="--")
    ax2.set_title("Rolling OLS beta of BTC to Nasdaq-100")
    ax2.set_ylabel("beta")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_business_cycle(res: BusinessCycle):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    panels = [
        (res.by_yield_curve, "BTC monthly return by yield-curve regime"),
        (res.by_growth, "BTC monthly return by industrial-production momentum"),
        (res.by_recession, "BTC monthly return: recession vs expansion"),
    ]
    for ax, (tbl, title) in zip(axes, panels):
        ax.bar(tbl.index, tbl["annualized_%"], color=["tab:green", "tab:red"][: len(tbl)])
        ax.set_title(title)
        ax.set_ylabel("annualized return %")
        ax.axhline(0, color="k", lw=0.8)
        for i, v in enumerate(tbl["annualized_%"]):
            ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom" if v >= 0 else "top")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
