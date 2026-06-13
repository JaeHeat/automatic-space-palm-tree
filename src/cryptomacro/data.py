"""High-level datasets built on top of the keyless FRED feed (:mod:`cryptomacro.fred`).

Everything here returns tidy pandas objects indexed by a ``DatetimeIndex`` named
``date`` so the analysis and plotting layers don't care where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .fred import fetch_series, fetch_series_chunked


def _robust_daily(series_id: str) -> pd.Series:
    """Fetch a daily series, falling back to chunked download on slow/flaky links."""
    try:
        return fetch_series(series_id)
    except Exception:  # noqa: BLE001 - large daily downloads can time out
        return fetch_series_chunked(series_id)

# ---------------------------------------------------------------------------
# Known event dates (UTC). Useful for cycle alignment / annotations.
# ---------------------------------------------------------------------------

HALVING_DATES = {
    "2012": pd.Timestamp("2012-11-28"),
    "2016": pd.Timestamp("2016-07-09"),
    "2020": pd.Timestamp("2020-05-11"),
    "2024": pd.Timestamp("2024-04-20"),
}

# Approximate BTC cycle price peaks (close-to-the-day tops).
CYCLE_PEAKS = {
    "2013": pd.Timestamp("2013-12-04"),
    "2017": pd.Timestamp("2017-12-17"),
    "2021": pd.Timestamp("2021-11-10"),
}


# ---------------------------------------------------------------------------
# Price series
# ---------------------------------------------------------------------------

def get_btc(start: str | None = None) -> pd.Series:
    """Daily BTC/USD (FRED ``CBBTCUSD``, Coinbase). History starts 2014-12-01."""
    s = fetch_series("CBBTCUSD")
    s.name = "BTC"
    return s.loc[start:] if start else s


def get_nasdaq100(start: str | None = None) -> pd.Series:
    """Daily Nasdaq-100 index level (FRED ``NASDAQ100``). A clean proxy for NQ futures."""
    s = _robust_daily("NASDAQ100")
    s.name = "NDX"
    return s.loc[start:] if start else s


# ---------------------------------------------------------------------------
# Global M2 liquidity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class M2Component:
    """One country/bloc's money supply plus how to convert it into USD.

    ``fx_series`` is a FRED exchange-rate series; ``fx_usd_per_unit`` says whether
    that series is quoted as USD-per-local (multiply) or local-per-USD (divide).
    ``scale`` brings the raw money-supply figure into absolute local-currency units.
    """

    name: str
    m2_series: str
    scale: float = 1.0
    fx_series: str | None = None
    fx_usd_per_unit: bool = True


# Default basket: US + Euro area + Japan "Broad Money" (OECD MABMM301 family).
#
# Data-source note: there is no single keyless FRED series family for global broad
# money that is current to today. The OECD "Broad Money" series (MABMM301*) are
# consistent across countries and run through ~2023-11; the older M2 series
# (MYAGM2*) stop in 2017-2019. We therefore build a consistent 3-bloc broad-money
# aggregate (US + Euro area + Japan) covering ~2014-2023, which spans the 2018
# bear, the 2020-21 bull and the 2022 bear -- enough to study regimes and lead-lag.
# China is available but its keyless series ends in 2018, so it is left out of the
# default basket (add CHINA_M2_COMPONENT to include it at the cost of a shorter window).
#
# FX uses *monthly* series (EX*) rather than daily (DEX*): M2 is monthly anyway.
DEFAULT_M2_BASKET: tuple[M2Component, ...] = (
    # MABMM301USM189S is broad money already in USD -> no FX, no scaling.
    M2Component("US", "MABMM301USM189S", fx_series=None),
    # Euro-area broad money in EUR; EXUSEU is USD per 1 EUR -> multiply.
    M2Component("EuroArea", "MABMM301EZM189S", fx_series="EXUSEU", fx_usd_per_unit=True),
    # Japan broad money in JPY; EXJPUS is JPY per 1 USD -> divide.
    M2Component("Japan", "MABMM301JPM189S", fx_series="EXJPUS", fx_usd_per_unit=False),
)

# Optional: China broad money (CNY). Keyless data ends 2018-12, so including it
# truncates the Global series to that date. EXCHUS is CNY per 1 USD -> divide.
CHINA_M2_COMPONENT = M2Component(
    "China", "MABMM301CNM189S", fx_series="EXCHUS", fx_usd_per_unit=False
)


def _to_monthly(s: pd.Series) -> pd.Series:
    """Resample any series to month-end, forward-filling within the month."""
    return s.resample("ME").last().ffill()


def get_global_m2(
    basket: tuple[M2Component, ...] = DEFAULT_M2_BASKET,
    *,
    start: str | None = None,
    in_trillions: bool = True,
) -> pd.DataFrame:
    """Build Global M2 in USD from a basket of national money-supply series.

    Returns a monthly DataFrame with one column per component plus a ``Global``
    total (each component converted to USD first). Values are in trillions of
    USD by default for readability.
    """
    components: dict[str, pd.Series] = {}
    for comp in basket:
        m2 = _to_monthly(fetch_series(comp.m2_series)) * comp.scale  # absolute local ccy
        if comp.fx_series is None:
            usd = m2
        else:
            fx = _to_monthly(fetch_series(comp.fx_series))
            usd = m2 * fx if comp.fx_usd_per_unit else m2 / fx
        components[comp.name] = usd

    df = pd.DataFrame(components).dropna(how="all")
    # Only sum where every component is present, so "Global" isn't biased by gaps.
    df["Global"] = df[[c.name for c in basket]].sum(axis=1, min_count=len(basket))
    if in_trillions:
        df = df / 1e12
    df = df.dropna(subset=["Global"])
    return df.loc[start:] if start else df


# ---------------------------------------------------------------------------
# Business-cycle indicators
# ---------------------------------------------------------------------------

# Keyless, regularly-updated proxies for the business / liquidity cycle.
# (ISM PMI is proprietary and not on FRED, so we use these instead.)
BUSINESS_CYCLE_SERIES = {
    "yield_curve_10y_2y": "T10Y2Y",   # >0 expansion-ish, <0 inversion warns of recession
    "industrial_production": "INDPRO",  # level; we use YoY growth downstream
    "unemployment_rate": "UNRATE",
    "nfci": "NFCI",                    # Chicago Fed financial conditions (>0 = tighter)
    "recession": "USREC",             # NBER recession flag (0/1)
}


def get_business_cycle(start: str | None = None) -> pd.DataFrame:
    """Monthly DataFrame of business-cycle indicators (see ``BUSINESS_CYCLE_SERIES``)."""
    # T10Y2Y is a long daily series; use the robust (chunked-fallback) fetch for it.
    def _fetch(sid: str) -> pd.Series:
        return _robust_daily(sid) if sid == "T10Y2Y" else fetch_series(sid)

    cols = {name: _to_monthly(_fetch(sid)) for name, sid in BUSINESS_CYCLE_SERIES.items()}
    df = pd.DataFrame(cols)
    df["industrial_production_yoy"] = df["industrial_production"].pct_change(12) * 100
    df = df.dropna(how="all")
    return df.loc[start:] if start else df
