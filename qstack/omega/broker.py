"""Broker interface and implementations.

The :class:`PaperBroker` is the default: it fills market orders against a price
you feed it (applying slippage + commission) and tracks cash, positions, and
realised equity entirely in process. The live brokers wrap ``ccxt`` and
``alpaca-py`` behind the same :class:`Broker` interface, so the workflow layer
never needs to know which one it is talking to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    type: str = "market"
    limit_price: float | None = None

    def __post_init__(self):
        self.side = Side(self.side)
        if self.qty <= 0:
            raise ValueError("order qty must be positive")


@dataclass
class Fill:
    order: Order
    price: float
    qty: float
    commission: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-12


class Broker:
    """Interface implemented by every backend."""

    def submit(self, order: Order, price: float | None = None) -> Fill:
        raise NotImplementedError

    def position(self, symbol: str) -> Position:
        raise NotImplementedError

    def equity(self, marks: dict[str, float] | None = None) -> float:
        raise NotImplementedError


class PaperBroker(Broker):
    """In-process fill simulator — the default execution backend."""

    def __init__(self, cash: float = 100_000.0, commission: float = 0.0005,
                 slippage: float = 0.0005):
        self.cash = float(cash)
        self.commission = commission
        self.slippage = slippage
        self._positions: dict[str, Position] = {}
        self.fills: list[Fill] = []

    def submit(self, order: Order, price: float | None = None) -> Fill:
        if price is None:
            if order.limit_price is None:
                raise ValueError("PaperBroker needs a price (pass price=... or a limit_price)")
            price = order.limit_price

        signed = 1 if order.side is Side.BUY else -1
        fill_price = price * (1 + signed * self.slippage)
        notional = fill_price * order.qty
        commission = notional * self.commission

        self.cash -= signed * notional + commission
        self._apply(order.symbol, signed * order.qty, fill_price)

        fill = Fill(order=order, price=fill_price, qty=order.qty, commission=commission)
        self.fills.append(fill)
        return fill

    def _apply(self, symbol: str, signed_qty: float, price: float):
        pos = self._positions.setdefault(symbol, Position(symbol))
        new_qty = pos.qty + signed_qty
        if pos.qty == 0 or (pos.qty > 0) == (signed_qty > 0):
            # opening or adding in the same direction -> blend the average price
            total = abs(pos.qty) + abs(signed_qty)
            pos.avg_price = (abs(pos.qty) * pos.avg_price + abs(signed_qty) * price) / total
        elif abs(signed_qty) > abs(pos.qty):
            pos.avg_price = price  # flipped through zero
        pos.qty = new_qty
        if pos.is_flat:
            pos.qty, pos.avg_price = 0.0, 0.0

    def position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol))

    def positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._positions.items() if not p.is_flat}

    def equity(self, marks: dict[str, float] | None = None) -> float:
        marks = marks or {}
        holdings = sum(p.qty * marks.get(s, p.avg_price) for s, p in self._positions.items())
        return self.cash + holdings


class CCXTBroker(Broker):  # pragma: no cover - requires credentials
    """Live crypto execution via ccxt."""

    def __init__(self, exchange="binance", api_key=None, secret=None):
        try:
            import ccxt
        except ImportError as exc:
            raise ImportError("ccxt not installed. Run: pip install 'qstack[crypto]'") from exc
        self.client = getattr(ccxt, exchange)({"apiKey": api_key, "secret": secret})

    def submit(self, order: Order, price: float | None = None) -> Fill:
        result = self.client.create_order(
            order.symbol, order.type, order.side.value, order.qty,
            order.limit_price if order.type == "limit" else None,
        )
        filled = result.get("average") or result.get("price") or price or 0.0
        return Fill(order=order, price=filled, qty=result.get("filled", order.qty),
                    commission=(result.get("fee") or {}).get("cost", 0.0))

    def position(self, symbol: str) -> Position:
        base = symbol.split("/")[0]
        bal = self.client.fetch_balance().get(base, {})
        return Position(symbol, qty=bal.get("total", 0.0))

    def equity(self, marks=None) -> float:
        return self.client.fetch_balance().get("total", {}).get("USDT", 0.0)


class AlpacaBroker(Broker):  # pragma: no cover - requires credentials
    """Live equities execution via alpaca-py (paper or live endpoint)."""

    def __init__(self, api_key=None, secret=None, paper=True):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError("alpaca-py not installed. Run: pip install 'qstack[broker]'") from exc
        self.client = TradingClient(api_key, secret, paper=paper)

    def submit(self, order: Order, price: float | None = None) -> Fill:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(
            symbol=order.symbol, qty=order.qty,
            side=OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        result = self.client.submit_order(req)
        return Fill(order=order, price=float(result.filled_avg_price or price or 0.0),
                    qty=float(result.filled_qty or order.qty), commission=0.0)

    def position(self, symbol: str) -> Position:
        try:
            p = self.client.get_open_position(symbol)
            return Position(symbol, qty=float(p.qty), avg_price=float(p.avg_entry_price))
        except Exception:
            return Position(symbol)

    def equity(self, marks=None) -> float:
        return float(self.client.get_account().equity)
