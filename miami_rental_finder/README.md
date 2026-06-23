# Miami Rental Finder 🌴

Pulls **for-rent** listings from Realtor.com (via the
[HomeHarvest](https://github.com/ZacharyHampton/HomeHarvest) package) for very
specific Miami areas, applies very specific filters, and alerts you about
**new** listings it hasn't shown you before.

Pre-configured for:

- **Kendall / Sunset / Kendall Drive** corridor
- **Coral Gables**
- **Brickell / Downtown**
- **Coconut Grove**

with filters for **price ceiling, beds/baths, square footage, and
pet-friendly / parking** keywords.

---

## Quick start

```bash
cd miami_rental_finder
pip install -r requirements.txt

python finder.py            # first run reports every current match
python finder.py            # later runs report only NEW listings
```

Outputs:

- `matches.csv` — every current match (overwritten each run)
- `new_matches.csv` — only the new listings from the latest run
- `seen.json` — bookkeeping so you only get alerted once per listing

---

## Configure it

Everything lives in **`config.yaml`** — no code changes needed.

### Areas
Each area is one query. A `location` can be a ZIP, `"Neighborhood, City, ST"`,
`"City, ST"`, or a full address with a `radius` (miles):

```yaml
areas:
  - { name: "Brickell",       location: "33131" }
  - { name: "Coral Gables",   location: "Coral Gables, FL" }
  - { name: "Near my office", location: "1 SE 3rd Ave, Miami, FL 33131", radius: 2.0 }
```

### Filters
Set any to `null` to disable:

```yaml
filters:
  price_max: 3200
  beds_min: 2
  baths_min: 2
  sqft_min: 900
  must_mention_any: [pet, dog, cat]   # listing text must mention one of these
  must_mention_all: [parking]         # ...and all of these
  exclude_if_mentions: ["no pets"]    # ...and none of these
```

> Structured fields (price, beds, baths, sqft) come straight from the listing.
> Amenities like *pet-friendly* and *parking* aren't always structured, so those
> are matched against the listing description text.

---

## Recurring alerts

You asked for **recurring new-listing alerts**. Three ways, easiest first.

### 1. Built-in loop
```bash
python finder.py --loop 30      # re-check every 30 minutes
```

### 2. cron (recommended for "set and forget")
```bash
# every 30 min, 8am–10pm; log output
*/30 8-22 * * *  cd /path/to/miami_rental_finder && /usr/bin/python3 finder.py >> finder.log 2>&1
```

### 3. Phone push via ntfy (free, no signup)
1. Install the **ntfy** app (iOS/Android).
2. Subscribe to a secret topic name, e.g. `miami-rentals-7f3q9`.
3. In `config.yaml`:
   ```yaml
   alerts:
     ntfy:
       enabled: true
       topic: "miami-rentals-7f3q9"
   ```
New matches now push straight to your phone. Email (SMTP/Gmail App Password) is
also supported — see the `alerts.email` block in `config.yaml`.

---

## Useful flags

```bash
python finder.py --reset            # forget seen listings (re-alert everything)
python finder.py --config other.yaml
```

---

## Notes & limits

- Data comes from Realtor.com via HomeHarvest; it can occasionally rate-limit.
  If a run returns 0 listings, wait and retry. Each area is queried separately
  so one failure won't sink the whole run.
- For personal use. Be respectful of source sites' terms; keep the schedule
  reasonable (every 15–30 min is plenty).
- Want Zillow too? HomeHarvest focuses on Realtor.com; a Zillow source could be
  added later behind the same filter pipeline.
