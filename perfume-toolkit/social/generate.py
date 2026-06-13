#!/usr/bin/env python3
"""
Generate ready-to-post social content — one pack per designer fragrance.

For each dupe it writes Instagram / TikTok captions, a short reel/voiceover
script, and a Canva-ready text brief, in English AND Spanish. Drop the text
into a post, or hand the brief to a designer / Canva.

Run:  python3 social/generate.py
Out:  social/posts/<slug>.md   (+ all_posts.md combined)
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "social", "posts")

HASHTAGS = ("#perfume #fragrance #perfumedupe #fragrancedupe #cologne #perfumelovers "
            "#designerdupe #miami #perfumetiktok #smellgood #fragancias #perfumes "
            "#perfumeria #miamiperfume #dupe")


def money(n):
    return "$" + ("%.0f" % n if float(n).is_integer() else "%.2f" % n)


def pack(d, dupe):
    best = dupe["products"][0]
    save = money(d["savings"]) if d["savings"] else ""
    en_notes = ", ".join(d["notes_en"])
    es_notes = ", ".join(d["notes_es"])
    hook_en = f'They\'ll think you\'re wearing {money(d["price"])} {d["designer"]}.'
    hook_es = f'Van a pensar que llevas {d["designer"]} de {money(d["price"])}.'
    return f"""# {d['designer']}  →  {best['title']}
**Confidence:** {d['confidence']}  ·  **Family:** {d['family']}  ·  **In-store from {money(d['from_price'])}** (retail {money(d['price'])}, save ~{save})
**Buy:** {best['url']}

---
## Instagram / Facebook caption — EN
{hook_en} 🤫

This is {best['title']} — the same {d['family'].lower()} vibe as {d['designer']}
({en_notes}) for just {money(best['price'])} instead of {money(d['price'])}. Save ~{save}. 💸

100% genuine. Free shipping over $75. DM us "{d['designer']}" to grab yours. 🛒
.
.
{HASHTAGS}

## Instagram / Facebook caption — ES
{hook_es} 🤫

Este es {best['title']} — el mismo aroma {d['family'].lower()} que {d['designer']}
({es_notes}) por solo {money(best['price'])} en vez de {money(d['price'])}. Ahorras ~{save}. 💸

100% original. Envío gratis +$75. Escríbenos "{d['designer']}" y es tuyo. 🛒
.
.
{HASHTAGS}

---
## TikTok / Reel script (15–20s)
- HOOK (0–3s): "Stop paying {money(d['price'])} for {d['designer']}." / "Deja de pagar {money(d['price'])} por {d['designer']}."
- REVEAL (3–8s): show {best['title']}, spray it. Text on screen: "{money(best['price'])} 😳"
- WHY (8–14s): list notes — {en_notes}. "Same {d['family'].lower()} energy."
- CTA (14–20s): "Link in bio · we ship from Miami 🌴 · 100% original"

## Canva brief (for a graphic / carousel)
- Slide 1: Big text "{d['designer']}" struck through → "{best['title']}"
- Slide 2: Price compare — {money(d['price'])} vs {money(best['price'])}, big green "SAVE {save}"
- Slide 3: Product photo + notes pills: {en_notes}
- Brand colors: deep brown/black background, gold accents. Logo bottom-right.
- Photo: {best['image']}

"""


def main():
    designers = json.load(open(os.path.join(DATA, "designers.json"), encoding="utf-8"))
    dupes = {d["slug"]: d for d in json.load(open(os.path.join(DATA, "dupes.json"), encoding="utf-8"))}
    os.makedirs(OUT, exist_ok=True)
    combined = []
    for d in designers:
        text = pack(d, dupes[d["slug"]])
        open(os.path.join(OUT, f'{d["slug"]}.md'), "w").write(text)
        combined.append(text)
    open(os.path.join(OUT, "all_posts.md"), "w").write(
        "# Felo Fragancias — Social content pack\n\n" + "\n\n".join(combined))
    print(f"generated {len(designers)} social packs + all_posts.md in social/posts/")


if __name__ == "__main__":
    main()
