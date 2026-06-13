#!/usr/bin/env python3
"""BTC vs Nasdaq-100: rolling correlation and rolling/full-sample OLS beta."""

import _bootstrap  # noqa: F401
from cryptomacro import analysis, plotting

if __name__ == "__main__":
    res = analysis.analyze_nq()
    print(res.summary)
    fig = plotting.plot_nq_correlation(res)
    out = _bootstrap.OUTPUTS / "nq_correlation.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved chart -> {out}")
