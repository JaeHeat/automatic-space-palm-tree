"""Fetch real daily OHLCV from Coinbase Exchange public API (no auth, no VPN).

Coinbase returns max 300 candles per request as [time, low, high, open, close, volume],
newest-first. We page backward in 300-day windows and cache each product to CSV.
"""
import csv
import os
import time
import json
import urllib.request
import datetime as dt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE = "https://api.exchange.coinbase.com/products/{pair}/candles"
START = dt.datetime(2022, 1, 1, tzinfo=dt.timezone.utc)
GRAN = 86400  # daily

# Major liquid coins with multi-year Coinbase history.
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
            "LTC-USD", "BCH-USD", "LINK-USD", "AVAX-USD", "DOT-USD", "XLM-USD"]


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return None


def fetch_product(pair):
    end = dt.datetime.now(dt.timezone.utc)
    rows = {}
    while end > START:
        start = max(START, end - dt.timedelta(days=300))
        url = (f"{BASE.format(pair=pair)}?granularity={GRAN}"
               f"&start={start.isoformat()}&end={end.isoformat()}")
        data = _get(url)
        if not data:
            break
        for t, low, high, op, close, vol in data:
            rows[t] = (t, op, high, low, close, vol)
        end = start
        time.sleep(0.34)  # stay under Coinbase public rate limit
    return [rows[k] for k in sorted(rows)]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for pair in PRODUCTS:
        rows = fetch_product(pair)
        if not rows:
            print(f"{pair}: NO DATA")
            continue
        path = os.path.join(DATA_DIR, f"{pair.replace('-', '_')}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for t, op, high, low, close, vol in rows:
                d = dt.datetime.fromtimestamp(t, dt.timezone.utc).date()
                w.writerow([d, op, high, low, close, vol])
        first = dt.datetime.fromtimestamp(rows[0][0], dt.timezone.utc).date()
        last = dt.datetime.fromtimestamp(rows[-1][0], dt.timezone.utc).date()
        print(f"{pair}: {len(rows)} days  {first} -> {last}")


if __name__ == "__main__":
    main()
