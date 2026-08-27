# Wildfire Site-Event Copilot (V1)

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
- Three Mireye API keys (round-robin rotation; this build assumes a Growth-plan key, 300
  requests/minute each)
- A NASA FIRMS `MAP_KEY` (free, register at https://firms.modaps.eosdis.nasa.gov/api/map_key/)
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
- **FIRMS MAP_KEY quota is UNVERIFIED** (per the SRS): the client degrades to a
  `firms_unavailable` flag on a quota signal rather than blocking the pipeline.
- **No ELMFIRE / delegated spread engine in V1.** The spread signal is a wind-projected ROS
  ellipse feature (a crude proxy, explicitly not Rothermel physics) - see
  `src/features/ros_ellipse.py`. The delegated operational engine is V2 scope.
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

`src/model/train.py` expects `.jsonl` samples in `data/training/`, one line per sample:

```json
{"site_id": "...", "event_id": "...", "t0": "2019-06-01T00:00:00Z",
 "w_vector": [...], "w_mask": [...], "e_vector": [...], "y": 0}
```

To build a real set (not synthetic):

1. Pick historical fires with a clean perimeter time series: MTBS final perimeters (public
   domain, ~1yr lag, large fires only) as the gold label source, or WFIGS final/YTD
   perimeters for smaller/recent fires.
2. For each `(site, event, t0)`, reconstruct **E as of t0** - FIRMS/WFIGS archive data and
   HRRR archive weather at that historical time, never today's values (that would leak the
   final outcome into the input).
3. Reconstruct **W at t0's vintage** via Mireye, masking any field (NDVI, drought category)
   that cannot be honestly backdated to that date; static fields (terrain, hazard zone
   geometry) are treated as stable.
4. Encode with `src/features/w_encoder.encode_w` and `src/features/e_packer.e_features_to_vector`
   to get `w_vector`/`w_mask`/`e_vector` in the exact same order the live pipeline uses.
5. Label `y = 1` if the site fell inside the fire's final perimeter within 72h of `t0`, else `0`.
6. Run `python scripts/train_model.py` - it evaluates on an event-held-out split and reports
   PR-AUC, Brier score, and the random-W collapse probe (SRS AC-7/AC-8: this is exactly what
   settles whether cited W actually earns its keep, or whether the system should ship as
   honest orchestration alone).

This is genuine ETL work spanning several public data sources plus per-site-per-date Mireye
calls; it was deliberately deferred so V1's codebase, tests, and pipeline could be complete
and correct first.
