#!/usr/bin/env python3
"""Compare BTC's current cycle to previous ones (halving-aligned overlay + drawdowns)."""

import _bootstrap  # noqa: F401  (sets sys.path / outputs)
from cryptomacro import analysis, plotting

if __name__ == "__main__":
    res = analysis.analyze_cycles()
    print(res.summary)
    fig = plotting.plot_cycle_comparison(res)
    out = _bootstrap.OUTPUTS / "cycle_comparison.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved chart -> {out}")
