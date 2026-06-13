#!/usr/bin/env python3
"""BTC monthly returns conditioned on US business-cycle regimes."""

import _bootstrap  # noqa: F401
from cryptomacro import analysis, plotting

if __name__ == "__main__":
    res = analysis.analyze_business_cycle()
    print(res.summary)
    fig = plotting.plot_business_cycle(res)
    out = _bootstrap.OUTPUTS / "business_cycle.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved chart -> {out}")
