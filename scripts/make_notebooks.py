#!/usr/bin/env python3
"""Generate the four analysis notebooks from a compact spec (no nbformat needed)."""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

BOOTSTRAP = (
    "import sys, pathlib\n"
    "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
    "%matplotlib inline\n"
    "import matplotlib; matplotlib.use('module://matplotlib_inline.backend_inline')\n"
    "from cryptomacro import analysis, plotting"
)

SPECS = {
    "01_cycle_comparison.ipynb": {
        "title": "# BTC cycle comparison\n\n"
                 "Overlay each post-halving cycle on a common axis (rebased to 100 at the "
                 "halving, log scale) and compare drawdown profiles. Useful for asking "
                 "*\"where are we vs the last bear/bull?\"*\n\n"
                 "> Data: BTC = FRED `CBBTCUSD` (history starts 2014-12, so the 2016/2020/2024 "
                 "halvings are covered).",
        "cells": [
            ("res = analysis.analyze_cycles()\nprint(res.summary)", "## Run"),
            ("res.stats", "Per-cycle stats:"),
            ("fig = plotting.plot_cycle_comparison(res); fig", "## Chart"),
        ],
    },
    "02_m2_liquidity.ipynb": {
        "title": "# BTC vs Global M2 liquidity\n\n"
                 "Global M2 = US + Euro area + China + Japan, each converted to USD. We "
                 "correlate **YoY growth rates** (not raw levels) to avoid spurious "
                 "trend-on-trend correlation, estimate the **lead-lag** (does liquidity "
                 "lead BTC?), and split correlation by **bull/bear regime**.",
        "cells": [
            ("res = analysis.analyze_m2_liquidity()\nprint(res.summary)", "## Run"),
            ("res.regime_corr", "Return correlation, bull vs bear:"),
            ("res.lead_lag.set_index('lag')", "Lead-lag table (months M2 leads BTC):"),
            ("fig = plotting.plot_m2_liquidity(res); fig", "## Chart"),
        ],
    },
    "03_nq_correlation.ipynb": {
        "title": "# BTC vs Nasdaq-100 (NQ)\n\n"
                 "Rolling correlation of daily returns plus a rolling and full-sample "
                 "**OLS beta** of BTC on the Nasdaq-100 — i.e. how much \"leveraged tech "
                 "risk\" BTC is carrying, and how that has drifted over time.",
        "cells": [
            ("res = analysis.analyze_nq()\nprint(res.summary)", "## Run"),
            ("res.full_ols", "Full-sample OLS (alpha/beta/R²/t-stats):"),
            ("fig = plotting.plot_nq_correlation(res); fig", "## Chart"),
        ],
    },
    "04_business_cycle.ipynb": {
        "title": "# BTC vs the business cycle\n\n"
                 "Condition BTC's monthly returns on US macro regimes: yield-curve "
                 "inversion, industrial-production momentum, and NBER recessions. Shows "
                 "average return, volatility and hit-rate in each regime.",
        "cells": [
            ("res = analysis.analyze_business_cycle()\nprint(res.summary)", "## Run"),
            ("res.by_yield_curve", "By yield-curve regime:"),
            ("res.by_growth", "By industrial-production momentum:"),
            ("fig = plotting.plot_business_cycle(res); fig", "## Chart"),
        ],
    },
}


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


def build(spec):
    cells = [md(spec["title"]), code(BOOTSTRAP)]
    for src, heading in spec["cells"]:
        if heading:
            cells.append(md(heading))
        cells.append(code(src))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    for fname, spec in SPECS.items():
        (NB_DIR / fname).write_text(json.dumps(build(spec), indent=1))
        print("wrote", fname)
