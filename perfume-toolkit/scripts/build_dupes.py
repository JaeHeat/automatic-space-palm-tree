#!/usr/bin/env python3
"""
Build the dupe-matching layer for Felo Fragancias.

Reads:  data/catalog.json   (scraped from the live Shopify store)
Writes: data/designers.json (reference list of designer fragrances)
        data/dupes.json      (designer fragrance -> matching in-store products)

HOW TO EXTEND (for the shop owner):
  Add a new entry to SEED below. `match` is a list of lowercase text
  fragments; any catalog product whose title contains one of them is linked
  to that designer fragrance. Re-run:  python3 scripts/build_dupes.py
"""
import json, os, re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

# --- Curated seed --------------------------------------------------------
# Each designer fragrance: the expensive original people search for, plus the
# in-store products that share its scent profile. Confidence reflects how
# widely the match is agreed on in the fragrance community.
# Language is deliberately "inspired by / smells similar to" — these are
# genuine branded perfumes, NOT counterfeits of the designer bottle.
SEED = [
    {
        "designer": "Creed Aventus", "house": "Creed", "price": 435,
        "gender": "men", "family": "Fruity Chypre",
        "notes_en": ["pineapple", "bergamot", "birch", "musk", "oakmoss"],
        "notes_es": ["piña", "bergamota", "abedul", "almizcle", "musgo de roble"],
        "match": ["club de nuit intense eau de toilette", "supremacy collector", "genius event"],
        "confidence": "high",
    },
    {
        "designer": "Creed Silver Mountain Water", "house": "Creed", "price": 400,
        "gender": "unisex", "family": "Fresh Woody",
        "notes_en": ["bergamot", "blackcurrant", "green tea", "musk"],
        "notes_es": ["bergamota", "grosella negra", "té verde", "almizcle"],
        "match": ["club de nuit sillage"],
        "confidence": "high",
    },
    {
        "designer": "Creed Green Irish Tweed", "house": "Creed", "price": 395,
        "gender": "men", "family": "Fresh Fougère",
        "notes_en": ["lemon verbena", "violet leaf", "sandalwood", "ambergris"],
        "notes_es": ["verbena", "hoja de violeta", "sándalo", "ámbar gris"],
        "match": ["tres nuit"],
        "confidence": "high",
    },
    {
        "designer": "Dior Sauvage", "house": "Dior", "price": 165,
        "gender": "men", "family": "Aromatic Fresh",
        "notes_en": ["bergamot", "pepper", "ambroxan", "lavender"],
        "notes_es": ["bergamota", "pimienta", "ambroxan", "lavanda"],
        "match": ["salvo", "club de nuit urban man elixir"],
        "confidence": "high",
    },
    {
        "designer": "Paco Rabanne Invictus", "house": "Paco Rabanne", "price": 135,
        "gender": "men", "family": "Aquatic Woody",
        "notes_en": ["grapefruit", "marine notes", "bay leaf", "guaiac wood"],
        "notes_es": ["pomelo", "notas marinas", "laurel", "madera de guayaco"],
        "match": ["victorioso"],
        "confidence": "high",
    },
    {
        "designer": "Bleu de Chanel", "house": "Chanel", "price": 175,
        "gender": "men", "family": "Woody Aromatic",
        "notes_en": ["grapefruit", "incense", "ginger", "cedar", "sandalwood"],
        "notes_es": ["pomelo", "incienso", "jengibre", "cedro", "sándalo"],
        "match": ["club de nuit urban man eau de parfum", "voyage bleu", "9 am dive", "ana al awwal blue"],
        "confidence": "medium",
    },
    {
        "designer": "Maison Francis Kurkdjian Baccarat Rouge 540", "house": "MFK", "price": 325,
        "gender": "unisex", "family": "Amber Floral",
        "notes_en": ["saffron", "jasmine", "ambergris", "cedar", "fir resin"],
        "notes_es": ["azafrán", "jazmín", "ámbar gris", "cedro", "resina de abeto"],
        "match": ["baroque la rouge", "ana abiyedh rouge", "club de nuit untold"],
        "confidence": "high",
    },
    {
        "designer": "Tom Ford Noir de Noir", "house": "Tom Ford", "price": 250,
        "gender": "unisex", "family": "Rose Oud",
        "notes_en": ["rose", "oud", "saffron", "patchouli", "vanilla"],
        "notes_es": ["rosa", "oud", "azafrán", "pachulí", "vainilla"],
        "match": ["jean lowe maître", "jean lowe maitre"],
        "confidence": "medium",
    },
    {
        "designer": "Kilian Angels' Share", "house": "Kilian", "price": 290,
        "gender": "unisex", "family": "Boozy Gourmand",
        "notes_en": ["cognac", "cinnamon", "tonka bean", "vanilla", "praline"],
        "notes_es": ["coñac", "canela", "haba tonka", "vainilla", "praliné"],
        "match": ["khamrah"],
        "confidence": "high",
    },
    {
        "designer": "Parfums de Marly Delina", "house": "Parfums de Marly", "price": 335,
        "gender": "women", "family": "Floral Fruity",
        "notes_en": ["lychee", "rose", "rhubarb", "vanilla", "musk"],
        "notes_es": ["lichi", "rosa", "ruibarbo", "vainilla", "almizcle"],
        "match": ["eclaire", "delilah"],
        "confidence": "medium",
    },
    {
        "designer": "Chanel Coco Mademoiselle", "house": "Chanel", "price": 165,
        "gender": "women", "family": "Floral Chypre",
        "notes_en": ["orange", "rose", "jasmine", "patchouli", "vanilla"],
        "notes_es": ["naranja", "rosa", "jazmín", "pachulí", "vainilla"],
        "match": ["club de nuit woman", "club de nuit intense woman"],
        "confidence": "medium",
    },
    {
        "designer": "Dior Sauvage Elixir", "house": "Dior", "price": 200,
        "gender": "men", "family": "Spicy Amber",
        "notes_en": ["cinnamon", "nutmeg", "grapefruit", "licorice", "amber"],
        "notes_es": ["canela", "nuez moscada", "pomelo", "regaliz", "ámbar"],
        "match": ["asad"],
        "confidence": "medium",
    },
    {
        "designer": "Creed Aventus (aquatic take)", "house": "Creed", "price": 435,
        "gender": "men", "family": "Fresh Fruity",
        "notes_en": ["apple", "bergamot", "ambergris", "musk", "lemon"],
        "notes_es": ["manzana", "bergamota", "ámbar gris", "almizcle", "limón"],
        "match": ["hawas for him"],
        "confidence": "medium",
    },
    {
        "designer": "Valentino Donna Born in Roma", "house": "Valentino", "price": 150,
        "gender": "women", "family": "Floral Woody",
        "notes_en": ["blackcurrant", "jasmine", "vanilla bourbon", "cashmeran"],
        "notes_es": ["grosella negra", "jazmín", "vainilla bourbon", "cashmeran"],
        "match": ["mia dolcezza"],
        "confidence": "medium",
    },
    {
        "designer": "Maison Francis Kurkdjian BR540 (sweet take)", "house": "MFK", "price": 325,
        "gender": "unisex", "family": "Sweet Amber",
        "notes_en": ["saffron", "jasmine", "amberwood", "caramel"],
        "notes_es": ["azafrán", "jazmín", "madera ambarada", "caramelo"],
        "match": ["bharara king"],
        "confidence": "medium",
    },
    {
        "designer": "Giorgio Armani Acqua di Giò Profumo", "house": "Armani", "price": 130,
        "gender": "men", "family": "Aquatic Aromatic",
        "notes_en": ["marine notes", "bergamot", "incense", "patchouli"],
        "notes_es": ["notas marinas", "bergamota", "incienso", "pachulí"],
        "match": ["turathi blue"],
        "confidence": "medium",
    },
    {
        "designer": "Jean Paul Gaultier Ultra Male", "house": "Jean Paul Gaultier", "price": 110,
        "gender": "men", "family": "Sweet Aromatic",
        "notes_en": ["pear", "lavender", "cinnamon", "vanilla", "amber"],
        "notes_es": ["pera", "lavanda", "canela", "vainilla", "ámbar"],
        "match": ["9 pm"],
        "confidence": "high",
    },
    {
        "designer": "Louis Vuitton Imagination", "house": "Louis Vuitton", "price": 350,
        "gender": "men", "family": "Citrus Tea Woody",
        "notes_en": ["bergamot", "black tea", "ginger", "ambrox", "neroli"],
        "notes_es": ["bergamota", "té negro", "jengibre", "ambrox", "neroli"],
        "match": ["jean lowe fantasme", "hawas cobra", "mazaaj infused"],
        "confidence": "high",
    },
    {
        "designer": "Yves Saint Laurent Y EDP", "house": "Yves Saint Laurent", "price": 120,
        "gender": "men", "family": "Aromatic Fougère",
        "notes_en": ["apple", "ginger", "sage", "geranium", "amberwood"],
        "notes_es": ["manzana", "jengibre", "salvia", "geranio", "madera ambarada"],
        "match": ["yeah! man"],
        "confidence": "medium",
    },
    {
        "designer": "Carolina Herrera Good Girl Blush", "house": "Carolina Herrera", "price": 130,
        "gender": "women", "family": "Floral Fruity Gourmand",
        "notes_en": ["jasmine", "tuberose", "vanilla", "tonka bean", "musk"],
        "notes_es": ["jazmín", "nardo", "vainilla", "haba tonka", "almizcle"],
        "match": ["yara"],
        "confidence": "medium",
    },
    {
        "designer": "Juliette Has a Gun Not a Perfume", "house": "Juliette Has a Gun", "price": 135,
        "gender": "unisex", "family": "Clean Musky Woody",
        "notes_en": ["ambroxan", "cetalox", "ambergris", "soft woods"],
        "notes_es": ["ambroxan", "cetalox", "ámbar gris", "maderas suaves"],
        "match": ["ana abiyedh eau de parfum"],
        "confidence": "medium",
    },
    {
        "designer": "Jean Paul Gaultier Scandal Pour Homme", "house": "Jean Paul Gaultier", "price": 95,
        "gender": "men", "family": "Sweet Aromatic",
        "notes_en": ["mandarin", "saffron", "caramel", "tonka bean", "vetiver"],
        "notes_es": ["mandarina", "azafrán", "caramelo", "haba tonka", "vetiver"],
        "match": ["odyssey mandarin sky"],
        "confidence": "medium",
    },
    {
        "designer": "Jean Paul Gaultier Le Male Elixir", "house": "Jean Paul Gaultier", "price": 115,
        "gender": "men", "family": "Sweet Amber",
        "notes_en": ["lavender", "honey", "vanilla", "tonka bean", "benzoin"],
        "notes_es": ["lavanda", "miel", "vainilla", "haba tonka", "benjuí"],
        "match": ["the kingdom"],
        "confidence": "medium",
    },
    {
        "designer": "Jean Paul Gaultier Le Beau Paradise Garden", "house": "Jean Paul Gaultier", "price": 100,
        "gender": "men", "family": "Tropical Coconut",
        "notes_en": ["coconut water", "fig", "ginger", "tonka bean"],
        "notes_es": ["agua de coco", "higo", "jengibre", "haba tonka"],
        "match": ["hawas tropical"],
        "confidence": "medium",
    },
    {
        "designer": "Guerlain Terracotta Le Parfum", "house": "Guerlain", "price": 110,
        "gender": "unisex", "family": "Tropical Floral",
        "notes_en": ["tiare flower", "coconut", "ylang-ylang", "vanilla", "neroli"],
        "notes_es": ["flor de tiaré", "coco", "ylang-ylang", "vainilla", "neroli"],
        "match": ["veneno bianco"],
        "confidence": "medium",
    },
    {
        "designer": "Narciso Rodriguez For Her", "house": "Narciso Rodriguez", "price": 130,
        "gender": "women", "family": "Floral Musk",
        "notes_en": ["rose", "musk", "amber", "patchouli", "vanilla"],
        "notes_es": ["rosa", "almizcle", "ámbar", "pachulí", "vainilla"],
        "match": ["narissa"],
        "confidence": "medium",
    },
    {
        "designer": "Louis Vuitton Nouveau Monde", "house": "Louis Vuitton", "price": 380,
        "gender": "unisex", "family": "Woody Spicy",
        "notes_en": ["saffron", "cacao", "agarwood", "patchouli", "leather"],
        "notes_es": ["azafrán", "cacao", "madera de agar", "pachulí", "cuero"],
        "match": ["jean lowe fraiche"],
        "confidence": "medium",
    },
]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def main():
    catalog = json.load(open(os.path.join(DATA, "catalog.json"), encoding="utf-8"))
    by_title = [(norm(p["title"]), p) for p in catalog]

    designers, dupes = [], []
    unmatched = []
    for s in SEED:
        prods = []
        seen = set()
        for frag in s["match"]:
            f = norm(frag)
            for t, p in by_title:
                if f in t and p["id"] not in seen:
                    seen.add(p["id"])
                    prods.append(p)
        if not prods:
            unmatched.append(s["designer"])
            continue
        prods.sort(key=lambda p: (p.get("price") or 9999))
        slug = re.sub(r"[^a-z0-9]+", "-", s["designer"].lower()).strip("-")
        cheapest = prods[0].get("price") or 0
        savings = round(s["price"] - cheapest) if cheapest else None
        designers.append({
            "slug": slug, "designer": s["designer"], "house": s["house"],
            "price": s["price"], "gender": s["gender"], "family": s["family"],
            "notes_en": s["notes_en"], "notes_es": s["notes_es"],
            "confidence": s["confidence"],
            "from_price": cheapest, "savings": savings,
            "match_count": len(prods),
        })
        dupes.append({
            "slug": slug, "designer": s["designer"],
            "products": [{
                "title": p["title"], "brand": p["brand"], "price": p["price"],
                "url": p["url"], "image": p["image"], "gender": p["gender"],
            } for p in prods],
        })

    json.dump(designers, open(os.path.join(DATA, "designers.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(dupes, open(os.path.join(DATA, "dupes.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"designers.json: {len(designers)} designer fragrances mapped")
    print(f"dupes.json:     {sum(len(d['products']) for d in dupes)} product links")
    if unmatched:
        print("WARNING unmatched seeds (no catalog product found):")
        for u in unmatched:
            print("   -", u)


if __name__ == "__main__":
    main()
