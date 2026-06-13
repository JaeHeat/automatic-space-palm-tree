"""High-level datasets built on top of the keyless FRED feed (:mod:`cryptomacro.fred`).

Everything here returns tidy pandas objects indexed by a ``DatetimeIndex`` named
``date`` so the analysis and plotting layers don't care where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .fred import fetch_series

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
    s = fetch_series("NASDAQ100")
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


# Default basket: US + Euro area + China + Japan. Together these dominate global M2.
DEFAULT_M2_BASKET: tuple[M2Component, ...] = (
    # M2SL is in *billions* of USD -> scale to absolute USD. No FX needed.
    M2Component("US", "M2SL", scale=1e9, fx_series=None),
    # Euro-area M2 in EUR; DEXUSEU is USD per 1 EUR -> multiply.
    M2Component("EuroArea", "MYAGM2EZM196N", fx_series="DEXUSEU", fx_usd_per_unit=True),
    # China M2 in CNY; DEXCHUS is CNY per 1 USD -> divide.
    M2Component("China", "MYAGM2CNM189N", fx_series="DEXCHUS", fx_usd_per_unit=False),
    # Japan M2 in JPY; DEXJPUS is JPY per 1 USD -> divide.
    M2Component("Japan", "MYAGM2JPM189S", fx_series="DEXJPUS", fx_usd_per_unit=False),
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
    cols = {name: _to_monthly(fetch_series(sid)) for name, sid in BUSINESS_CYCLE_SERIES.items()}
    df = pd.DataFrame(cols)
    df["industrial_production_yoy"] = df["industrial_production"].pct_change(12) * 100
    df = df.dropna(how="all")
    return df.loc[start:] if start else df
