"""Omega equivalent — execute trades with Python.

A small broker abstraction so research and live trading share one interface:

    Order        a normalised order (symbol, side, qty, type, price)
    Broker       the interface: submit / positions / equity
    PaperBroker  in-process fill simulator (default; no credentials, no network)
    CCXTBroker   live crypto execution via ccxt          (extra: qstack[crypto])
    AlpacaBroker live equities execution via alpaca-py    (extra: qstack[broker])

Swap PaperBroker for a live broker without touching strategy or workflow code.
"""

from qstack.omega.broker import (
    AlpacaBroker,
    Broker,
    CCXTBroker,
    Fill,
    Order,
    PaperBroker,
    Position,
)

__all__ = [
    "Order",
    "Fill",
    "Position",
    "Broker",
    "PaperBroker",
    "CCXTBroker",
    "AlpacaBroker",
]
