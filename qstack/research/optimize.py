"""Parameter optimization — grid-search a strategy's parameters.

``grid_search`` runs a backtest for every combination in a parameter grid and
returns a ranked table of results, so you can see how robust a strategy is to
its settings rather than trusting a single hand-picked configuration.

    from qstack.research import grid_search, sma_crossover
    table = grid_search(df, sma_crossover, {"fast": [10, 20], "slow": [50, 100]})
    table.head()                      # best configs first (by Sharpe)
"""

from __future__ import annotations

import itertools
from typing import Callable

import pandas as pd

from qstack.research.backtest import Backtest


def grid_search(
    df: pd.DataFrame,
    strategy: Callable[..., pd.Series],
    param_grid: dict[str, list],
    *,
    backtest: Backtest | None = None,
    rank_by: str = "sharpe",
    ascending: bool = False,
    valid: Callable[[dict], bool] | None = None,
) -> pd.DataFrame:
    """Backtest ``strategy`` across every combination in ``param_grid``.

    Args:
        df:          OHLCV frame to test on.
        strategy:    callable ``(df, **params) -> position series``.
        param_grid:  mapping of parameter name -> list of values to try.
        backtest:    engine to use (defaults to ``Backtest()`` with standard costs).
        rank_by:     stat column to sort the results table by.
        ascending:   sort direction (default: best Sharpe first).
        valid:       optional predicate to drop nonsensical combos, e.g.
                     ``lambda p: p["fast"] < p["slow"]``.

    Returns:
        DataFrame with one row per parameter combination: the parameter columns
        plus every backtest stat, sorted by ``rank_by``.
    """
    engine = backtest or Backtest()
    names = list(param_grid)
    rows = []
    for combo in itertools.product(*(param_grid[n] for n in names)):
        params = dict(zip(names, combo))
        if valid is not None and not valid(params):
            continue
        result = engine.run(df, strategy(df, **params))
        rows.append({**params, **result.stats})

    if not rows:
        raise ValueError("grid_search produced no valid parameter combinations")

    table = pd.DataFrame(rows)
    if rank_by not in table.columns:
        raise ValueError(f"rank_by={rank_by!r} not in stats {sorted(set(table.columns) - set(names))}")
    return table.sort_values(rank_by, ascending=ascending).reset_index(drop=True)
