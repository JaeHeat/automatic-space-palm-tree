"""cryptomacro: macro/crypto data analysis on a single keyless data source (FRED).

Four analyses, all reusable from scripts or notebooks:

1. analysis.analyze_cycles()          -- current cycle vs previous bears/bulls
2. analysis.analyze_m2_liquidity()    -- BTC vs Global M2, bull/bear + lead-lag
3. analysis.analyze_nq()              -- BTC vs Nasdaq-100, rolling corr + OLS beta
4. analysis.analyze_business_cycle()  -- BTC returns conditioned on the business cycle
"""

from . import analysis, data, fred, plotting, transforms

__all__ = ["analysis", "data", "fred", "plotting", "transforms"]
__version__ = "0.1.0"
