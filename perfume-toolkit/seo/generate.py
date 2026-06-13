#!/usr/bin/env python3
"""
Generate SEO landing pages — one per designer fragrance.

Targets the searches dupe shoppers actually type:
  "<designer> dupe", "perfume that smells like <designer>",
  "<designer> alternative", "clon de <designer>".

Each page is static HTML with proper <title>/meta/OpenGraph + schema.org
Product markup, so Google can index and rank it. English + Spanish versions.

Run:  python3 seo/generate.py
Out:  seo/pages/<slug>.html and seo/pages/es/<slug>.html  (+ index.html)
"""
import json, os, re, html

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "seo", "pages")
SITE = "https://felofraganciasshop.com"

COPY = {
    "en": {
        "title": "{d} Dupe — Smells Like It for {from_} | Felo Fragancias",
        "meta": "Looking for a {d} dupe? This {fam} fragrance smells just like {d} for {from_} instead of {price} — save about {save}. Genuine, in stock, ships from Miami.",
        "h1": "The {d} dupe that actually smells like it",
        "lead": "Love {d} but not the {price} price tag? These genuine fragrances share the same {fam} character — saffron-to-musk, the notes people recognize — for as little as {from_}.",
        "notesh": "Why it smells like {d}",
        "picks": "Our picks", "save": "Save ~{save}", "vs": "retail",
        "shop": "Shop now", "browse": "See all fragrance matches",
        "faqq": "Is this a copy of {d}?",
        "faqa": "No. These are authentic, branded perfumes from respected fragrance houses. They are built around a similar scent profile to {d}. We are not affiliated with {house}.",
        "indexh": "Find your designer fragrance match",
        "indexsub": "Smells like the perfume you love — at a fraction of the price.",
    },
    "es": {
        "title": "Clon de {d} — Huele Igual por {from_} | Felo Fragancias",
        "meta": "¿Buscas un clon de {d}? Esta fragancia {fam} huele igualito a {d} por {from_} en vez de {price} — ahorra unos {save}. Original, en stock, envíos desde Miami.",
        "h1": "El clon de {d} que sí huele igual",
        "lead": "¿Te encanta {d} pero no el precio de {price}? Estas fragancias originales comparten el mismo carácter {fam} — las notas que todos reconocen — desde apenas {from_}.",
        "notesh": "Por qué huele a {d}",
        "picks": "Nuestras opciones", "save": "Ahorra ~{save}", "vs": "precio original",
        "shop": "Comprar", "browse": "Ver todas las coincidencias",
        "faqq": "¿Es una copia de {d}?",
        "faqa": "No. Son perfumes originales de marca, de casas de fragancias reconocidas, construidos alrededor de un perfil de aroma similar a {d}. No estamos afiliados con {house}.",
        "indexh": "Encuentra tu fragancia de diseñador",
        "indexsub": "Huele al perfume que amas — por una fracción del precio.",
    },
}

PAGE = """<!DOCTYPE html><html lang="{lang}"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{meta}">
<meta property="og:title" content="{title}"><meta property="og:description" content="{meta}">
<meta property="og:image" content="{ogimg}"><meta property="og:type" content="website">
<link rel="canonical" href="{canonical}">
<link rel="alternate" hreflang="en" href="{alt_en}"><link rel="alternate" hreflang="es" href="{alt_es}">
<script type="application/ld+json">{schema}</script>
<style>
body{{margin:0;font-family:"Helvetica Neue",Arial,sans-serif;background:#0e0b08;color:#f3ece0;line-height:1.6}}
.w{{max-width:820px;margin:0 auto;padding:34px 18px 70px}}
a{{color:#e7c977;text-decoration:none}}
.bk{{letter-spacing:.14em;text-transform:uppercase;font-size:12px;color:#c9a24b}}
h1{{font-size:30px;line-height:1.18;margin:14px 0}}
.lead{{color:#cdbfa6;font-size:18px}}
.save{{display:inline-block;background:#15321f;color:#7fe0a0;border:1px solid #28432f;padding:4px 11px;border-radius:999px;font-size:13px;font-weight:700;margin:6px 0}}
h2{{font-size:18px;margin:34px 0 10px;color:#e7c977}}
.notes{{display:flex;flex-wrap:wrap;gap:7px}}
.note{{border:1px solid #2c2218;border-radius:999px;padding:4px 11px;font-size:13px;color:#cdbfa6}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:8px}}
.card{{background:#1a1410;border:1px solid #2c2218;border-radius:14px;overflow:hidden;display:flex;flex-direction:column}}
.card .img{{aspect-ratio:1;background:#fff center/cover no-repeat}}
.card .b{{padding:12px 13px}}
.card .t{{font-size:14px;font-weight:600;margin:4px 0}}
.card .p{{font-size:17px;font-weight:700}}
.card .vs{{font-size:12px;color:#a7977f;text-decoration:line-through;margin-left:6px}}
.cta{{display:block;text-align:center;background:#c9a24b;color:#1a1206;font-weight:700;padding:9px;border-radius:9px;margin-top:8px}}
.faq{{background:#14100b;border:1px solid #2c2218;border-radius:12px;padding:16px 18px;margin-top:26px}}
.foot{{margin-top:40px;font-size:12px;color:#6f6450;line-height:1.7}}
</style></head><body><div class="w">
<div class="bk">Felo Fragancias · Miami</div>
<h1>{h1}</h1>
<p class="lead">{lead}</p>
<div class="save">{savebadge}</div>
<h2>{notesh}</h2>
<div class="notes">{notes}</div>
<h2>{picks}</h2>
<div class="grid">{cards}</div>
<div class="faq"><h2 style="margin-top:0">{faqq}</h2><p>{faqa}</p></div>
<p style="margin-top:24px"><a href="{finder}">→ {browse}</a></p>
<div class="foot">{legal}</div>
</div></body></html>"""


def money(n):
    return "$" + ("%.0f" % n if float(n).is_integer() else "%.2f" % n)


def card(p, vs, lang, c):
    vsh = f'<span class="vs">{money(vs)} {c["vs"]}</span>' if vs else ""
    return f"""<div class="card"><div class="img" style="background-image:url('{html.escape(p['image'] or '')}')"></div>
<div class="b"><div class="bk">{html.escape(p['brand'])}</div><div class="t">{html.escape(p['title'])}</div>
<div><span class="p">{money(p['price'])}</span>{vsh}</div>
<a class="cta" href="{html.escape(p['url'])}">{c['shop']}</a></div></div>"""


def render(lang, d, dupe):
    c = COPY[lang]
    notes = d["notes_es"] if lang == "es" else d["notes_en"]
    f = dict(d=d["designer"], house=d["house"], fam=d["family"].lower(),
             price=money(d["price"]), from_=money(d["from_price"]),
             save=money(d["savings"]) if d["savings"] else "")
    cards = "".join(card(p, d["price"], lang, c) for p in dupe["products"])
    base = SITE  # change to your hosted path if deploying standalone
    canonical = f'{base}/dupe/{d["slug"]}.html' if lang == "en" else f'{base}/dupe/es/{d["slug"]}.html'
    schema = json.dumps({
        "@context": "https://schema.org", "@type": "Product",
        "name": f'{d["designer"]} alternative fragrance',
        "description": c["meta"].format(**f),
        "brand": {"@type": "Brand", "name": "Felo Fragancias"},
        "offers": {"@type": "AggregateOffer", "lowPrice": d["from_price"],
                   "highPrice": d["price"], "priceCurrency": "USD",
                   "offerCount": d["match_count"]},
    }, ensure_ascii=False)
    return PAGE.format(
        lang=lang, title=html.escape(c["title"].format(**f)),
        meta=html.escape(c["meta"].format(**f)),
        ogimg=dupe["products"][0]["image"] or "", canonical=canonical,
        alt_en=f'{base}/dupe/{d["slug"]}.html', alt_es=f'{base}/dupe/es/{d["slug"]}.html',
        schema=schema, h1=html.escape(c["h1"].format(**f)),
        lead=html.escape(c["lead"].format(**f)),
        savebadge=c["save"].format(**f) if d["savings"] else "",
        notesh=html.escape(c["notesh"].format(**f)),
        notes="".join(f'<span class="note">{html.escape(n)}</span>' for n in notes),
        picks=c["picks"], cards=cards,
        faqq=html.escape(c["faqq"].format(**f)), faqa=html.escape(c["faqa"].format(**f)),
        finder="../finder/index.html" if lang == "en" else "../../finder/index.html",
        browse=c["browse"],
        legal="All products are genuine, branded fragrances. Designer names describe similar scent profiles only; we are not affiliated with the designer brands.")


def main():
    designers = json.load(open(os.path.join(DATA, "designers.json"), encoding="utf-8"))
    dupes = {d["slug"]: d for d in json.load(open(os.path.join(DATA, "dupes.json"), encoding="utf-8"))}
    os.makedirs(os.path.join(OUT, "es"), exist_ok=True)
    n = 0
    for d in designers:
        dupe = dupes[d["slug"]]
        open(os.path.join(OUT, f'{d["slug"]}.html'), "w").write(render("en", d, dupe))
        open(os.path.join(OUT, "es", f'{d["slug"]}.html'), "w").write(render("es", d, dupe))
        n += 2
    # simple index of all pages
    links = "".join(
        f'<li><a href="{d["slug"]}.html">{html.escape(d["designer"])}</a> '
        f'— from {money(d["from_price"])} '
        f'(<a href="es/{d["slug"]}.html">ES</a>)</li>' for d in designers)
    open(os.path.join(OUT, "index.html"), "w").write(
        f'<!DOCTYPE html><meta charset="utf-8"><title>Designer fragrance matches</title>'
        f'<body style="font-family:Arial;background:#0e0b08;color:#f3ece0;max-width:700px;margin:40px auto;padding:0 16px">'
        f'<h1 style="color:#e7c977">{COPY["en"]["indexh"]}</h1>'
        f'<p style="color:#a7977f">{COPY["en"]["indexsub"]}</p><ul style="line-height:2">{links}</ul>')
    print(f"generated {n} landing pages + index in seo/pages/")


if __name__ == "__main__":
    main()
