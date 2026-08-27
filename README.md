# Wildfire Site-Event Copilot (V1 + V2 spread)

An unattended, cited fire-week copilot for a book of named US sites. While a wildfire is
live or imminent near a site, it joins live fire signals (NWS Red Flag alerts, NASA FIRMS
thermal hotspots, WFIGS incidents and operational perimeters, HRRR wind/weather), cited
site physics from the Mireye API (fuels, terrain, hazard zone, access, burn history), and a
trained model into a closed, cited ActionCard: monitor, prepare, protect_asset,
evacuate_site, inspect_after, or no_action. V2 adds an incident-first AOI, LANDFIRE fuel
rasters, an out-of-process spread engine (`spread_run`) that emits ETA / P(burn by T) with
ensemble sigma, ETA-keyed policy, OSM truck routing, and an AOI water map. When a site
escalates to protect_asset or evacuate_site, a second Response Support Agent produces a
ResponseCard: ranked water sources, access routes, fire-station ETA, hazmat priorities,
environmental constraints, and responsible agency, all deterministic and cited, no trained
model involved. The action is always decided by a versioned policy table, never the LLM;
the LLM only writes a copy-only prose brief, which is rejected and replaced if it
introduces a single number not already on the card.

An unattended, cited fire-week copilot for a book of named US sites. While a wildfire is
live or imminent near a site, it joins live fire signals (NWS Red Flag alerts, NASA FIRMS
thermal hotspots, WFIGS incidents and operational perimeters, HRRR wind/weather), cited
site physics from the Mireye API (fuels, terrain, hazard zone, access, burn history), and a
trained model into a closed, cited ActionCard: monitor, prepare, protect_asset,
evacuate_site, inspect_after, or no_action. When a site escalates to protect_asset or
evacuate_site, a second Response Support Agent produces a ResponseCard: ranked water
sources, access routes, fire-station ETA, hazmat priorities, environmental constraints, and
responsible agency, all deterministic and cited, no trained model involved. The action is
always decided by a versioned policy table, never the LLM; the LLM only writes a copy-only
prose brief, which is rejected and replaced if it introduces a single number not already on
the card.

This is a decision tool for a known book of commercial/industrial sites, not a public safety
broadcaster. It never downgrades or suppresses an NWS warning and never substitutes for an
evacuation order.

## Prerequisites

- Python 3.11+ (built and tested on 3.12.13)
- Two or three Mireye API keys (round-robin; this build assumes a Growth-plan key, 300
  requests/minute each)
- A NASA FIRMS `MAP_KEY` (free, register at https://firms.modaps.eosdis.nasa.gov/api/map_key/;
  optional - the pipeline degrades with `firms_unavailable` if unset)
- An OpenAI API key (for the copy-only brief; the system runs without one, using a
  deterministic fallback brief)
- Optionally: a Slack bot token and SMTP credentials for real delivery (both delivery
  channels run in log-only mode without them - see `config/README_MISSING.md`)

## Installation

```bash
git clone <this repo>
cd fire_copilot
python -m venv .venv
.venv/Scripts/activate   # or source .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
cp .env.example .env
# fill in MIREYE_KEY_1/2/3, OPENAI_KEY, FIRMS_MAP_KEY in .env
```

## Quick start

```bash
python scripts/generate_synthetic_training_data.py
python scripts/train_model.py
python scripts/watch_loop.py
```

The synthetic data generator and trainer bootstrap a real, versioned `h_fire_v*.pkl` model
artifact so the watch loop has something to call from the first run, before any historical
fire data has been collected (see "Known limitations" below - this model has no real skill
yet, it exists so the pipeline is runnable end to end).

## Ask mode

```bash
python scripts/ask.py --lat 33.98 --lng -117.37 --q "Is the site at risk?"
```

Runs the full pipeline once for that coordinate: fetches live E and cited W, scores it,
applies the policy, prints the ActionCard JSON and its brief, and - if the action is
`protect_asset` or `evacuate_site` - the ResponseCard JSON as well.

## Config

- `config/sites.yaml` - the book of sites for watch mode. Add a site with `site_id`, `name`,
  `lat`/`lng` (or `address`, geocoded via Mireye), `slack_channel`, `email`.
- `config/policy.yaml` - every tunable policy threshold (distance cutoffs, y_hat cutoffs,
  sigma suppression, dedup window, hazmat tiers, fire-station speed assumptions). Change
  numbers here, not in code, and bump `version` when you do - it is stamped onto every
  ActionCard as `policy_version`.
- `config/field_sets.yaml` - the ~60-field Mireye Fire Intelligence Set, grouped by role
  (A-J per the SRS), with each field's encoding type and (for categoricals) its real
  taxonomy, verified against the live API.

## Output

- `data/logs/YYYY-MM-DD.jsonl` - append-only log of every tool call (request/response),
  model call, policy decision, and credit spend. Never contains a raw API key, only a key
  index. This is the audit trail: every card is reconstructable from these logs.
- `data/models/h_fire_v*.pkl` + `data/models/metrics_*.json` - trained model artifacts and
  their evaluation metrics (PR-AUC, Brier score, random-W collapse probe).
- `data/cache/w_cache.db` - the SQLite W cache (role-dependent TTL: 30 days for static
  roles, 3-14 days for vintage-dynamic roles depending on fire season).
- `data/cache/site_state.db` - per-site state (last poll, last action, last delivery) used
  for watch-loop idempotency and de-duplication.

## Credit usage

```bash
python scripts/credit_report.py --days 30
```

Sums the `quote`-before-`fetch` credit log entries and reports spend against the V1
quality envelope (~300,000-400,000 credits, SRS NFR-6). Credits are not rationed - this is
a visibility tool, not a spending cap.

## Architecture

```
book of sites (config/sites.yaml)
        |
        v
  +-----------------------------------------------------------+
  |                     MAIN AGENT (watch | ask)               |
  |  geocode -> nws_alerts -> [firms | wfigs | wfigs  | hrrr]  |
  |                            hotspots  incidents perimeters  |  (parallel)
  |         -> pack_e -> ros_ellipse -> quote+fetch W (cached) |
  |         -> encode_w -> model_infer -> policy -> ActionCard |
  |         -> LLM brief -> brief_validator -> deliver          |
  +-----------------------------------------------------------+
        |                                   |
        | (protect_asset / evacuate_site)   |
        v                                   v
  +----------------------+          Slack / email (or log-only)
  | RESPONSE AGENT        |
  | role-J W + live USGS  |
  | gage -> water/access/ |
  | hazmat/constraints/   |
  | agency/comms -> Card  |
  +----------------------+
        |
        v
  Slack thread reply / email (or log-only)
```

`model_infer`, the ActionCard schema, and the policy contract are stable; everything behind
`model_infer` (currently a small MLP + isotonic calibration) can be swapped for a richer
model without touching the agent.

## Known limitations

- **The shipped model has no real skill yet.** `data/models/h_fire_v*.pkl` is trained on
  `scripts/generate_synthetic_training_data.py`'s 500 samples with **random 50/50 labels**,
  purely so the pipeline (train -> infer -> policy -> card) runs end to end before real
  historical fire data has been collected. Its PR-AUC and random-W collapse numbers are not
  a research result - see "How to collect training data" below for what real training needs.
- ~~FIRMS MAP_KEY quota is UNVERIFIED~~ - fixed 2026-08-27: it's a real, documented,
  checkable limit (`GET /mapserver/mapkey_status/?MAP_KEY=...` returns 5,000
  transactions/10 minutes). `FIRMSClient` self-throttles against it and exposes
  `get_quota_status()` for a live check; the `firms_unavailable` degraded-flag path is kept
  as defense in depth, not because the limit is unknown.
- **No compiled ELMFIRE binary in this tree.** V2 still runs a delegated spread engine
  *out of process* (`spread_service/`, `POST /spread_run`). The process currently uses an
  original Rothermel+Huygens solver fed by LANDFIRE FBFM40 + HRRR; set `SPREAD_ENGINE_BIN`
  to exec a real elmfire wrapper that speaks the JSON file contract. See `DECISIONS.md`.
  V1's wind-projected ROS ellipse remains as an E-side feature.
- **H2 is not yet evaluable on real fires.** `data/training/` has no MTBS/LANDFIRE/spread
  JSONL in this checkout. `python scripts/evaluate_h2.py` reports
  `unevaluable_no_real_spread_labels` and runs a synthetic probe only. That is the honest
  AC-13-adjacent outcome until `--with-spread` collection exists.
- **HRRR's grid is Lambert Conformal** (2D curvilinear lat/lon), so nearest-point lookup is
  done by brute-force distance argmin, not `xarray.sel(method="nearest")` - see
  `src/clients/hrrr.py` and `DECISIONS.md`.
- **Mireye has no waterbody/flowline distance field**, only the name - `ResponseCard`'s
  `WaterSource.distance_m` is `null` for those two source types rather than a guess.
- **SPC's Day-1 fire weather outlook has no reliable plain-text feed.** The client points at
  the real serving HTML page and does best-effort keyword extraction, which is biased
  toward false negatives (never a fabricated "elevated" reading). This is an explicitly
  low-priority slow signal (SRS 2.2.1) and never blocks the pipeline either way.
- Slack and email delivery run in log-only mode until `SLACK_BOT_TOKEN`/`SMTP_*` are set in
  `.env` (see `config/README_MISSING.md`).

## How to collect real training data

This is implemented, not just described - `scripts/build_training_set.py` builds real
samples from MTBS's final-perimeter database (30,000+ fires nationally, public domain):

```bash
python scripts/build_training_set.py \
  --bbox -125 24 -66 49 \
  --year-start 2010 --year-end 2023 \
  --min-acres 1000 \
  --max-fires 1000 \
  --positives-per-fire 3 --hard-negatives-per-fire 3 --easy-negatives-per-fire 2 \
  --credit-target 290000 \
  --output data/training/real_conus_2010_2023.jsonl
```

For each matching fire it samples points inside the final perimeter (positive), 2-20 km
outside it (hard negative), and far from any known fire (easy negative, per SRS 6.2), then
builds one real sample per point: a real Mireye `fetch` for W, a real FIRMS-archive query
for historical hotspots at that date, and a real HRRR-archive fetch for wind at that hour.
`--credit-target` stops the run once cumulative Mireye spend hits a target (credits are
priced at 1/field/location, confirmed live, so cost is exact, not estimated). Add
`--with-spread` to fetch LANDFIRE once per fire and attach a delegated `spread_vector`
for the H2 calibration head. Then:

```bash
python scripts/train_model.py
python scripts/evaluate_h2.py
```

**Read `src/model/training_data.py`'s module docstring before trusting the labels for a
real H1 claim.** The honest limitation: MTBS publishes only the fire's ignition *date* and
*final* perimeter, not a perimeter time series, so `dist_perim_m` here is distance to the
fire's ignition centroid, not "distance to the perimeter as it existed at t0" (SRS 6.3's
actual definition) - using the *final* perimeter's distance would leak the outcome (SRS
6.1's leakage rule), so the weaker ignition-centroid proxy is used instead. `acres` and
`containment_pct` are left null for the same reason, not backfilled from the final MTBS
acreage. Vintage-dynamic W fields (`ndvi_current`, `ndvi_change_5y`, `drought_category`)
are stripped before encoding since Mireye only serves today's value for those, never a
historical one (SRS 6.2: "never use 2026 NDVI on a 2018 fire"). A stronger version of this
pipeline would reconstruct t0 from a real historical WFIGS/NIFC perimeter-snapshot archive
instead of proxying off MTBS's ignition date alone - that is the next real improvement, not
yet built.

Once a real set exists at `data/training/*.jsonl`, run:

```bash
python scripts/train_model.py
```

It evaluates on an event-held-out split and reports PR-AUC, Brier score, and the random-W
collapse probe (SRS AC-7/AC-8: this is exactly what settles whether cited W earns its keep,
or whether the system should ship as honest orchestration alone).
