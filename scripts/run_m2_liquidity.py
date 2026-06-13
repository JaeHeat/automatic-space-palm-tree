#!/usr/bin/env python3
"""BTC vs Global M2 liquidity: rolling correlation, lead-lag, bull/bear regimes."""

import _bootstrap  # noqa: F401
from cryptomacro import analysis, plotting

if __name__ == "__main__":
    res = analysis.analyze_m2_liquidity()
    print(res.summary)
    fig = plotting.plot_m2_liquidity(res)
    out = _bootstrap.OUTPUTS / "m2_liquidity.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved chart -> {out}")
