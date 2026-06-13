# Felo Fragancias — Growth Toolkit

Three tools that all run off **one shared dataset** scraped from the live store.
Built for [felofraganciasshop.com](https://felofraganciasshop.com): 593 real
products, bilingual EN/ES, "smells like a designer perfume for a fraction of
the price."

```
perfume-toolkit/
├── data/
│   ├── catalog.json     # 593 products (title, brand, price, notes, image, url)
│   ├── designers.json   # designer fragrances people search for
│   └── dupes.json       # designer  →  matching in-store products
├── finder/index.html    # 1. Dupe-finder web app (bilingual, embeddable)
├── seo/generate.py      # 2. SEO landing pages (one per designer, EN+ES)
├── social/generate.py   # 3. Social captions + reel scripts + Canva briefs
└── scripts/
    ├── build_catalog.py # re-scrape the store -> catalog.json
    └── build_dupes.py   # edit dupe matches -> designers.json + dupes.json
```

## The big idea
Every dupe shopper asks the same thing: *"what do you have that smells like
[expensive perfume]?"* This toolkit answers that question — on the website, on
Google, and on social — automatically.

---

## 1. Dupe-finder app  (`finder/index.html`)
Customer types a designer perfume → sees the in-store bottle that smells like it,
with the price they save. Falls back to note-matching for anything not curated.

**Preview locally:**
```bash
cd perfume-toolkit && python3 -m http.server 8000
# open http://localhost:8000/finder/
```

**Put it on the Shopify store** (easiest options):
- Create a page, add a *Custom Liquid* / *Embed* section with an iframe:
  ```html
  <iframe src="https://YOUR-HOST/finder/?data=https://YOUR-HOST/data"
          style="width:100%;height:900px;border:0"></iframe>
  ```
- Host the `finder/` + `data/` folders anywhere static (GitHub Pages, Netlify,
  Cloudflare Pages — all free). The `?data=` param points the app at the JSON.
- Link it from the menu as **"Find your match" / "Encuentra tu aroma"** and put
  the link in the Instagram/TikTok bio.

---

## 2. SEO landing pages  (`seo/generate.py`)
One indexable page per designer fragrance, targeting real searches like
*"baccarat rouge 540 dupe"* and *"clon de sauvage"*. Includes schema.org
markup, OpenGraph, and EN + ES versions with hreflang.

```bash
python3 seo/generate.py   # -> seo/pages/*.html and seo/pages/es/*.html
```

Publish these as blog posts / pages on Shopify, or host the `seo/pages/` folder
and link to product pages. This is how dupe shops pull in free Google traffic.

---

## 3. Social content packs  (`social/generate.py`)
For each dupe: Instagram/Facebook captions, a 15–20s TikTok/reel script, and a
Canva graphic brief — in English and Spanish.

```bash
python3 social/generate.py   # -> social/posts/*.md and social/posts/all_posts.md
```

Open `social/posts/all_posts.md`, copy a block, post it. The Canva brief tells
you (or a designer) exactly what graphic to make.

---

## Keeping it fresh / extending it

**Re-scrape the store** (after adding products):
```bash
python3 scripts/build_catalog.py
```

**Add or fix a dupe match** — edit the `SEED` list in `scripts/build_dupes.py`
(`match` = text fragments found in product titles), then:
```bash
python3 scripts/build_dupes.py && python3 seo/generate.py && python3 social/generate.py
```

The seed currently covers **16 high-confidence matches**. Your dad-in-law knows
the catalog best — adding more is a few lines each, and everything (finder, SEO,
social) updates from that one edit.

## Important — keep it legal & honest
Every product here is a **genuine, branded perfume** (Lattafa, Armaf, Maison
Alhambra, etc.). The designer names are used only to describe a *similar scent
profile*. All copy says **"inspired by / smells similar to"**, never "fake" or
"copy of", and includes a disclaimer of non-affiliation. Keep it that way.
