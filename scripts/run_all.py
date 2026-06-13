#!/usr/bin/env python3
"""Run all four analyses, print summaries, and save every chart to outputs/."""

import _bootstrap  # noqa: F401
from cryptomacro import analysis, plotting

ANALYSES = [
    ("cycle_comparison", analysis.analyze_cycles, plotting.plot_cycle_comparison),
    ("m2_liquidity", analysis.analyze_m2_liquidity, plotting.plot_m2_liquidity),
    ("nq_correlation", analysis.analyze_nq, plotting.plot_nq_correlation),
    ("business_cycle", analysis.analyze_business_cycle, plotting.plot_business_cycle),
]

if __name__ == "__main__":
    for name, run, plot in ANALYSES:
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        res = run()
        print(res.summary)
        fig = plot(res)
        out = _bootstrap.OUTPUTS / f"{name}.png"
        fig.savefig(out, dpi=120)
        print(f"Saved chart -> {out}")
