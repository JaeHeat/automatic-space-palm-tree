#!/usr/bin/env python3
"""Pre-warm the FRED cache one series at a time, with spacing + retries.

Run this once before the analyses if your network/proxy rate-limits bursts of
requests. After it completes, all four analyses run from the local cache.
"""

import time

import _bootstrap  # noqa: F401
from cryptomacro.fred import fetch_series

SERIES = [
    "CBBTCUSD",            # BTC
    "NASDAQ100",           # Nasdaq-100
    "M2SL",                # US M2
    "MYAGM2EZM196N",       # Euro-area M2 (EUR)
    "MYAGM2CNM189N",       # China M2 (CNY)
    "MYAGM2JPM189S",       # Japan M2 (JPY)
    "DEXUSEU", "DEXCHUS", "DEXJPUS",   # FX
    "T10Y2Y", "INDPRO", "UNRATE", "NFCI", "USREC",  # business cycle
]

if __name__ == "__main__":
    ok, fail = [], []
    for sid in SERIES:
        try:
            s = fetch_series(sid, use_cache=True)
            print(f"  {sid:16s} {len(s):6d} obs  {s.index.min().date()}..{s.index.max().date()}")
            ok.append(sid)
        except Exception as e:  # noqa: BLE001
            print(f"  {sid:16s} FAILED: {e}")
            fail.append(sid)
        time.sleep(1.5)  # be gentle with the egress / FRED
    print(f"\nDone. {len(ok)} ok, {len(fail)} failed.")
    if fail:
        print("Failed:", ", ".join(fail), "-- re-run prefetch to retry these.")
