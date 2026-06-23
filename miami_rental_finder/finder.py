#!/usr/bin/env python3
"""
Miami Rental Finder
-------------------
Pulls for-rent listings from Realtor.com (via the HomeHarvest package) for a set
of configured Miami areas, applies your filters, and alerts you about *new*
listings it hasn't reported before.

Usage:
    python finder.py                  # one run using config.yaml
    python finder.py --config my.yaml # use a different config
    python finder.py --loop 30        # keep running, re-check every 30 minutes
    python finder.py --reset          # forget previously-seen listings

The first run reports every current match. After that, only listings that are
new since the previous run are sent as alerts (and written to new_matches.csv).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cfg_get(d: dict, *keys, default=None):
    """Safely walk a nested dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


# --------------------------------------------------------------------------- #
# Scraping
# --------------------------------------------------------------------------- #
def scrape_areas(areas: list[dict], past_days: int) -> pd.DataFrame:
    """Query each configured area and return a combined, de-duplicated frame."""
    try:
        from homeharvest import scrape_property
    except ImportError:
        sys.exit(
            "HomeHarvest is not installed. Run:\n"
            "    pip install -r requirements.txt"
        )

    frames = []
    for area in areas:
        name = area.get("name", area.get("location", "?"))
        location = area["location"]
        radius = area.get("radius")
        try:
            kwargs = dict(
                location=location,
                listing_type="for_rent",
                past_days=past_days,
            )
            if radius:
                kwargs["radius"] = float(radius)
            df = scrape_property(**kwargs)
        except Exception as e:  # one bad area shouldn't kill the whole run
            print(f"  ! {name} ({location}): query failed: {e}")
            continue

        if df is None or len(df) == 0:
            print(f"  - {name} ({location}): 0 listings")
            continue

        df = df.copy()
        df["area_name"] = name
        frames.append(df)
        print(f"  - {name} ({location}): {len(df)} listings")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # De-duplicate the same unit appearing in overlapping searches.
    key = "property_url" if "property_url" in combined.columns else None
    if key:
        combined = combined.drop_duplicates(subset=[key]).reset_index(drop=True)
    return combined


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def total_baths(df: pd.DataFrame) -> pd.Series:
    full = _num(df["full_baths"]) if "full_baths" in df else 0
    half = _num(df["half_baths"]) if "half_baths" in df else 0
    full = full.fillna(0) if hasattr(full, "fillna") else full
    half = half.fillna(0) if hasattr(half, "fillna") else half
    return full + 0.5 * half


def _text_blob(df: pd.DataFrame) -> pd.Series:
    """Concatenate the text-ish columns so keyword filters can scan them."""
    cols = [c for c in ("text", "description", "style", "primary_photo")
            if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    blob = df[cols[0]].astype(str)
    for c in cols[1:]:
        blob = blob.str.cat(df[c].astype(str), sep=" ")
    return blob.str.lower()


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)

    price = _num(df["list_price"]) if "list_price" in df else None
    if price is not None:
        if f.get("price_max") is not None:
            mask &= price.le(f["price_max"]) | price.isna()
        if f.get("price_min") is not None:
            mask &= price.ge(f["price_min"]) | price.isna()

    if "beds" in df:
        beds = _num(df["beds"])
        if f.get("beds_min") is not None:
            mask &= beds.ge(f["beds_min"])
        if f.get("beds_max") is not None:
            mask &= beds.le(f["beds_max"])

    baths = total_baths(df)
    if f.get("baths_min") is not None:
        mask &= baths.ge(f["baths_min"])

    if "sqft" in df:
        sqft = _num(df["sqft"])
        if f.get("sqft_min") is not None:
            mask &= sqft.ge(f["sqft_min"]) | sqft.isna()
        if f.get("sqft_max") is not None:
            mask &= sqft.le(f["sqft_max"]) | sqft.isna()

    blob = _text_blob(df)
    any_kw = [k.lower() for k in (f.get("must_mention_any") or [])]
    all_kw = [k.lower() for k in (f.get("must_mention_all") or [])]
    excl_kw = [k.lower() for k in (f.get("exclude_if_mentions") or [])]

    if any_kw:
        mask &= blob.apply(lambda t: any(k in t for k in any_kw))
    if all_kw:
        mask &= blob.apply(lambda t: all(k in t for k in all_kw))
    if excl_kw:
        mask &= blob.apply(lambda t: not any(k in t for k in excl_kw))

    return df[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# State (seen listings)
# --------------------------------------------------------------------------- #
def listing_id(row: pd.Series) -> str:
    for col in ("property_url", "mls_id", "listing_id"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    # Fallback: synthesize an id from address-ish fields.
    parts = [str(row.get(c, "")) for c in ("street", "unit", "zip_code", "list_price")]
    return "|".join(parts)


def load_seen(path: Path) -> set[str]:
    if path.exists():
        try:
            return set(json.loads(path.read_text()).get("seen", []))
        except Exception:
            return set()
    return set()


def save_seen(path: Path, seen: set[str]) -> None:
    path.write_text(json.dumps({"seen": sorted(seen),
                                "updated": datetime.now().isoformat()}, indent=2))


# --------------------------------------------------------------------------- #
# Presentation + alerts
# --------------------------------------------------------------------------- #
DISPLAY_COLS = [
    "area_name", "list_price", "beds", "full_baths", "half_baths", "sqft",
    "street", "unit", "city", "zip_code", "property_url",
]


def describe(row: pd.Series) -> str:
    price = row.get("list_price")
    price_s = f"${int(price):,}/mo" if pd.notna(price) else "$?/mo"
    beds = row.get("beds")
    beds_s = f"{int(beds)}bd" if pd.notna(beds) else "?bd"
    baths = total_baths(pd.DataFrame([row])).iloc[0]
    baths_s = f"{baths:g}ba" if pd.notna(baths) else "?ba"
    sqft = row.get("sqft")
    sqft_s = f"{int(sqft)} sqft" if pd.notna(sqft) else "? sqft"
    addr = " ".join(str(row.get(c, "")) for c in ("street", "unit", "city")).strip()
    area = row.get("area_name", "")
    url = row.get("property_url", "")
    return f"[{area}] {price_s} · {beds_s}/{baths_s} · {sqft_s} · {addr}\n  {url}"


def notify_ntfy(cfg: dict, title: str, body: str) -> None:
    import requests
    server = cfg.get("server", "https://ntfy.sh").rstrip("/")
    topic = cfg.get("topic")
    if not topic:
        return
    try:
        requests.post(
            f"{server}/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": "default", "Tags": "house"},
            timeout=15,
        )
    except Exception as e:
        print(f"  ! ntfy notification failed: {e}")


def notify_email(cfg: dict, subject: str, body: str) -> None:
    import smtplib
    from email.mime.text import MIMEText

    user = cfg.get("username") or os.environ.get("RENTAL_SMTP_USER", "")
    pw = cfg.get("password") or os.environ.get("RENTAL_SMTP_PASS", "")
    to_addrs = cfg.get("to_addrs") or []
    from_addr = cfg.get("from_addr") or user
    if not (user and pw and to_addrs):
        print("  ! email enabled but username/password/to_addrs missing; skipping")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    try:
        with smtplib.SMTP(cfg["smtp_host"], int(cfg.get("smtp_port", 587))) as s:
            s.starttls()
            s.login(user, pw)
            s.sendmail(from_addr, to_addrs, msg.as_string())
    except Exception as e:
        print(f"  ! email notification failed: {e}")


def send_alerts(new_df: pd.DataFrame, alerts_cfg: dict) -> None:
    if new_df.empty:
        return
    lines = [describe(r) for _, r in new_df.iterrows()]
    body = "\n\n".join(lines)
    title = f"{len(new_df)} new Miami rental(s)"

    if cfg_get(alerts_cfg, "ntfy", "enabled", default=False):
        notify_ntfy(alerts_cfg["ntfy"], title, body)
    if cfg_get(alerts_cfg, "email", "enabled", default=False):
        notify_email(alerts_cfg["email"], title, body)


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run_once(config: dict, base: Path) -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== Run @ {stamp} ===")

    areas = config.get("areas", [])
    past_days = cfg_get(config, "search", "past_days", default=7)

    print("Scraping areas...")
    raw = scrape_areas(areas, past_days)
    if raw.empty:
        print("No listings returned. (Realtor.com may be rate-limiting; try again later.)")
        return 0

    matches = apply_filters(raw, config.get("filters", {}))
    print(f"{len(matches)} of {len(raw)} listings match your filters.")

    out = config.get("output", {})
    matches_csv = base / out.get("matches_csv", "matches.csv")
    new_csv = base / out.get("new_csv", "new_matches.csv")
    seen_path = base / out.get("seen_state", "seen.json")

    keep = [c for c in DISPLAY_COLS if c in matches.columns]
    extra = [c for c in matches.columns if c not in keep]
    matches[keep + extra].to_csv(matches_csv, index=False)

    seen = load_seen(seen_path)
    ids = matches.apply(listing_id, axis=1) if not matches.empty else pd.Series([], dtype=str)
    is_new = ~ids.isin(seen) if len(ids) else pd.Series([], dtype=bool)
    new_df = matches[is_new].reset_index(drop=True) if len(ids) else matches

    if len(new_df):
        new_df[keep + extra].to_csv(new_csv, index=False)
        print(f"\n{len(new_df)} NEW listing(s):\n")
        for _, r in new_df.iterrows():
            print(describe(r) + "\n")
        send_alerts(new_df, config.get("alerts", {}))
    else:
        print("No new listings since last run.")

    seen.update(ids.tolist())
    save_seen(seen_path, seen)
    return len(new_df)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Miami rental finder with new-listing alerts.")
    ap.add_argument("--config", default="config.yaml", help="Path to config file")
    ap.add_argument("--loop", type=float, metavar="MINUTES",
                    help="Keep running, re-checking every N minutes")
    ap.add_argument("--reset", action="store_true",
                    help="Forget previously-seen listings, then run")
    args = ap.parse_args()

    base = Path(args.config).resolve().parent
    config = load_config(Path(args.config))

    if args.reset:
        seen_path = base / cfg_get(config, "output", "seen_state", default="seen.json")
        if seen_path.exists():
            seen_path.unlink()
            print(f"Cleared {seen_path}")

    if args.loop:
        print(f"Looping every {args.loop} minutes. Ctrl-C to stop.")
        try:
            while True:
                run_once(config, base)
                time.sleep(args.loop * 60)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        run_once(config, base)


if __name__ == "__main__":
    main()
