"""Vectorized backtester.

Pure numpy/pandas so it always runs. Positions are applied with a one-bar lag (a
signal computed on bar *t*'s close is traded into at bar *t+1*), which avoids
look-ahead bias. Fees and slippage are charged on traded notional whenever the
position changes.

If ``vectorbt`` is installed you can call :meth:`Backtest.run_vectorbt` for the
same semantics on its optimized engine; the default :meth:`run` needs nothing
beyond the core deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series          # cumulative equity curve, starts at 1.0
    returns: pd.Series         # per-bar strategy returns (net of costs)
    positions: pd.Series       # applied position per bar (after lag)
    stats: dict                # summary performance metrics

    def __repr__(self) -> str:
        lines = ["BacktestResult("]
        for k, v in self.stats.items():
            lines.append(f"    {k:<16} {v:>10.4f}")
        return "\n".join(lines) + "\n)"

    def summary(self) -> pd.Series:
        return pd.Series(self.stats)


class Backtest:
    def __init__(self, fee: float = 0.0005, slippage: float = 0.0005,
                 periods_per_year: int = TRADING_DAYS):
        self.fee = fee
        self.slippage = slippage
        self.periods_per_year = periods_per_year

    def run(self, df: pd.DataFrame, signal: pd.Series) -> BacktestResult:
        price = df["close"].astype(float)
        bar_ret = price.pct_change().fillna(0.0)

        # Trade on the *next* bar to avoid look-ahead.
        position = signal.reindex(price.index).ffill().fillna(0.0).shift(1).fillna(0.0)

        trades = position.diff().abs().fillna(position.abs())
        cost = trades * (self.fee + self.slippage)
        strat_ret = position * bar_ret - cost

        equity = (1 + strat_ret).cumprod()
        return BacktestResult(
            equity=equity.rename("equity"),
            returns=strat_ret.rename("returns"),
            positions=position.rename("position"),
            stats=self._stats(strat_ret, equity, trades),
        )

    def run_vectorbt(self, df: pd.DataFrame, signal: pd.Series):  # pragma: no cover
        """Run the same idea on vectorbt when the optional extra is installed."""
        try:
            import vectorbt as vbt
        except ImportError as exc:
            raise ImportError(
                "vectorbt not installed. Run: pip install 'qstack[backtest]'"
            ) from exc
        position = signal.reindex(df.index).ffill().fillna(0.0)
        entries = (position > 0) & (position.shift(1) <= 0)
        exits = (position <= 0) & (position.shift(1) > 0)
        return vbt.Portfolio.from_signals(
            df["close"], entries, exits, fees=self.fee, slippage=self.slippage,
            freq="1D",
        )

    def _stats(self, ret: pd.Series, equity: pd.Series, trades: pd.Series) -> dict:
        n = len(ret)
        if n == 0 or equity.iloc[-1] <= 0:
            return {k: 0.0 for k in
                    ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "n_trades")}

        years = n / self.periods_per_year
        total_return = equity.iloc[-1] - 1.0
        cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
        vol = ret.std()
        sharpe = (ret.mean() / vol * np.sqrt(self.periods_per_year)) if vol > 0 else 0.0
        drawdown = (equity / equity.cummax() - 1).min()
        active = ret[trades > 0]
        win_rate = float((ret > 0).sum() / max((ret != 0).sum(), 1))

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
            "win_rate": win_rate,
            "n_trades": float((trades > 0).sum()),
        }
