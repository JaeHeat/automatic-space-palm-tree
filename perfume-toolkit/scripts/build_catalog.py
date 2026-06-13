#!/usr/bin/env python3
"""
Scrape the live Shopify store into data/catalog.json.

Shopify exposes the full product list at /products.json (paginated, 250/page).
This pulls every page, cleans each product down to what the toolkit needs,
and writes data/catalog.json.

Run:  python3 scripts/build_catalog.py
"""
import json, os, re, urllib.request

STORE = "https://felofraganciasshop.com"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

NON_NOTE = {"eau de parfum", "eau de toilette", "femenino", "masculino",
            "unisex", "men", "women", "white", "black", "gift set", "set"}


def clean(h):
    return re.sub(r"\s+", " ", re.sub("<[^>]+>", "", h or "")).strip()


def gender(tags):
    t = [x.lower() for x in tags]
    if "unisex" in t:
        return "unisex"
    if "masculino" in t or "men" in t:
        return "men"
    if "femenino" in t or "women" in t:
        return "women"
    return "unisex"


def fetch_all():
    prods = []
    for page in range(1, 11):
        url = f"{STORE}/products.json?limit=250&page={page}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.load(urllib.request.urlopen(req, timeout=30)).get("products", [])
        if not data:
            break
        prods += data
        print(f"  page {page}: {len(data)} products")
    return prods


def main():
    os.makedirs(DATA, exist_ok=True)
    catalog = []
    for p in fetch_all():
        price = None
        for v in (p.get("variants") or []):
            try:
                price = float(v.get("price"))
                break
            except (TypeError, ValueError):
                pass
        tags = p.get("tags") or []
        notes = [t for t in tags if t.lower() not in NON_NOTE and not t.startswith("#")]
        imgs = p.get("images") or []
        catalog.append({
            "id": p["id"], "title": p["title"].strip(),
            "brand": (p.get("vendor") or "").strip(),
            "handle": p["handle"], "url": f"{STORE}/products/{p['handle']}",
            "price": price, "gender": gender(tags), "notes": notes,
            "desc": clean(p.get("body_html"))[:600],
            "image": imgs[0]["src"] if imgs else "",
        })
    json.dump(catalog, open(os.path.join(DATA, "catalog.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"wrote catalog.json with {len(catalog)} products")


if __name__ == "__main__":
    main()
