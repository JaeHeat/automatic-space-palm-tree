#!/usr/bin/env python3
"""Pre-warm the FRED cache, with spacing + retries.

Run this once before the analyses if your network/proxy rate-limits bursts of
requests or is slow. Monthly series are fetched whole; large *daily* series
(Nasdaq-100, the yield curve) are fetched in one-year chunks so each request
stays small. After it completes, all four analyses run from the local cache.
"""

import time

import _bootstrap  # noqa: F401
from cryptomacro.fred import fetch_series, fetch_series_chunked

# Monthly / small series -> single request each.
MONTHLY = [
    "CBBTCUSD",                                    # BTC (daily but compact)
    "MABMM301USM189S", "MABMM301EZM189S", "MABMM301JPM189S",  # broad money US/EU/JP
    "EXUSEU", "EXJPUS",                            # monthly FX
    "INDPRO", "UNRATE", "NFCI", "USREC",           # business cycle
]

# Large daily series -> fetch in yearly chunks.
DAILY_CHUNKED = ["NASDAQ100", "T10Y2Y"]

if __name__ == "__main__":
    ok, fail = [], []
    for sid in MONTHLY:
        try:
            s = fetch_series(sid)
            print(f"  {sid:18s} {len(s):6d} obs  {s.index.min().date()}..{s.index.max().date()}")
            ok.append(sid)
        except Exception as e:  # noqa: BLE001
            print(f"  {sid:18s} FAILED: {e}")
            fail.append(sid)
        time.sleep(1.0)

    for sid in DAILY_CHUNKED:
        try:
            s = fetch_series_chunked(sid)
            print(f"  {sid:18s} {len(s):6d} obs  {s.index.min().date()}..{s.index.max().date()} (chunked)")
            ok.append(sid)
        except Exception as e:  # noqa: BLE001
            print(f"  {sid:18s} FAILED: {e}")
            fail.append(sid)

    print(f"\nDone. {len(ok)} ok, {len(fail)} failed.")
    if fail:
        print("Failed:", ", ".join(fail), "-- re-run prefetch to retry these.")
