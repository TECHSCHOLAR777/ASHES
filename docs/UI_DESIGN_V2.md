# ASHES V2 UI design

Design for the V2 web UI. V1 screens in `docs/UI_DESIGN_V1.md` stay. V2 changes what
appears **behind** the same ActionCard / ResponseCard contract: arrival time, ensemble
sigma, incident AOI, LANDFIRE-fed spread field, AOI water map, OSM truck ETA.

Do not rebuild the watch board from scratch. Add the surfaces below, and swap V1's
score-primary layout for an ETA-primary layout when a spread field exists.

## 1. What changed for a human

| V1 UI | V2 UI |
|---|---|
| Primary number: `y_hat` = P(in perimeter 72 h) | Primary number: `eta_hours` ± `eta_sigma_hours`, labeled **arrival estimate with sigma** (FR-28) |
| Baseline: distance + ROS ellipse | Baseline: raw delegated `p_burn_72` when a field exists |
| Point pin + optional WFIGS ring | Incident AOI polygon, burnable-fuel clip, arrival raster |
| Fire-station ETA = distance / 60 or 30 km/h | OSM road-network ETA when Overpass returns a graph; else the V1 proxy, labeled |
| Water list at the site point | Same list **plus** a spatially referenced AOI water map (max 25 Mireye cells) |
| Policy reasons from distance / y_hat | Same table plus ETA 24-72 h prepare and ETA < 6 h protect/evacuate |
| No spread version | Footer always shows `spread_field_version` when a run happened |

If `eta_hours` is null (site outside the field, no perimeter, or LANDFIRE/`spread_run`
failed), the V1 score layout is the fallback. Hide empty ETA rather than showing "0 h".

## 2. Design principles (added)

1. **Never call the raster "the perimeter."** Copy is "arrival estimate with sigma" or
   "P(burn by T), ensemble". Official perimeters remain WFIGS operational polygons and
   stay flagged unofficial.
2. **High sigma is a first-class UI state.** If `no_ros_high_sigma` is set, the evacuate
   pill must not look like a confident evac. Show "downgraded to protect (high sigma)"
   from `reasons`.
3. **Null ETA is not a failed UI.** A live LA ask produced a real
   `spread_field_version` (`rothermel_huygens_v1:n7:h72:grid174x412`) with ETA null
   because the site was 113 km from the incident. That is correct. Render the field on
   the map; leave the ETA clock empty.
4. **Cost mental model:** one spread run per active incident, reused across nearby
   sites (NFR-4). The UI may say "field shared with N sites" but must not imply a
   per-site physics solve.
5. All V1 null / citation / no-invented-number rules still apply.

## 3. Information architecture (delta)

```
App  (V1 routes unchanged)
├── Watch board          + ETA column, AOI badge, field-version chip
├── Site                 + field history (spread_field_version over time)
├── Card                 + ETA clock, P(burn) bars, AOI/spread map
├── Ask                  + tool step `spread_run` after W fetch
├── Response dossier     + water map, OSM route line, eta_source badge
├── Incident             /incidents/:irwin_id     NEW: one field, many sites
└── Audit                + landfire:* and spread_run tool rows
```

New: **Incident** page. V2 cost and geometry are incident-first (FR-4, FR-9). Clicking
an IRWIN id from any card opens the shared AOI + arrival field, with the book of sites
drawn as pins (in-AOI vs out).

## 4. Watch board delta

Insert an ETA column after Distance:

```
│ Site         │ Action   │ ETA (σ)        │ Distance   │ y_hat     │
│ Riverside    │ PREPARE  │ 36 h ± 4 h     │ 8.2 km     │ 0.31±0.12 │
│ Warehouse    │          │ arrival est.   │ IR-1       │ vs 0.22   │
│ Flagstaff    │ MONITOR  │ — (outside     │ 113 km     │ 0.10±0.10 │
│ Timber Yard  │          │    field)      │ LAC-301937 │           │
```

- If `eta_hours` is set: show hours and `eta_sigma_hours`. Caption under the cell:
  "arrival estimate with sigma".
- If `prepare_min_hours` (24) ≤ ETA ≤ `prepare_max_hours` (72): optional small "prepare
  window" tag from `config/policy.yaml`.
- If ETA < `evacuate_hours` (6): treat as imminent in the row (does not override the
  action pill; policy already did).
- If null: em dash is **not** allowed if it could be read as zero. Use "outside field"
  when `spread_field_version` is set, or "no field" when the run never happened
  (`degraded` / no WFIGS perimeter).
- AOI badge on the site name when `site.mode == "aoi"`.

Sort: imminent ETA first, then V1 escalation order.

## 5. Card delta (ActionCard V2)

### 5.1 Decision band

Same action pill. When ETA drove the action, `reasons` already contain "ETA X h is
inside the 24-72 h prepare window" or "ETA X h (arrival estimate with sigma)". Surface
those first.

When `no_ros_high_sigma` is present, a banner:

> Evacuate was suppressed because uncertainty is too high
> (`eta_sigma_hours` and/or `sigma`). Action is `protect_asset`.

### 5.2 Score band (ETA-primary)

Show this block **only if** `eta_hours` is not null **or** `p_burn_by_T` has any
non-null horizon.

```
┌──────────────────────────────────────────────────────────┐
│ Arrival estimate with sigma                              │
│  14 h  ±  3 h                                            │
│  P(burn by T)   24h ████░░░░  0.41                       │
│                 48h ██████░░  0.67                       │
│                 72h ███████░  0.81                       │
│  Raw field P(burn 72 h)  0.81   (baseline_y)             │
│  Calibrated y_hat        0.74 ± 0.18                     │
│  spread_field_version  rothermel_huygens_v1:n7:h72:...   │
│  policy_version v2.0.0                                   │
└──────────────────────────────────────────────────────────┘
```

Rules:

- Horizons with null P(burn) are omitted, not drawn as 0.
- `baseline_y` in V2 is the raw delegated p_burn_72 when the field sampled the site.
  Label it "raw spread field", not "distance+ROS", in that case.
- `y_hat` remains P(burn / in-perimeter 72 h) from `model_infer`. If the artefact is
  untrained, keep the V1 untrained banner.
- Do not animate the bars as if they were live video. They are a single ensemble snapshot.

If ETA is null but `spread_field_version` is set, show a compact line:

> Spread field `rothermel_huygens_v1:...` ran for this incident. This site was not
> inside a reached cell, so no arrival estimate is shown.

### 5.3 Map band (new)

Replace V1's pin-only mini-map with a two-layer map on the Card and Incident pages.

**Layers (all optional, toggles default on when data exists):**

| Layer | Source | Style |
|---|---|---|
| Site pin | `site.lat/lng` | Black dot, label name |
| Book of sites | `config/sites.yaml` intersect AOI | Hollow dots; filled if `mode==aoi"` |
| WFIGS perimeter | operational rings | Magenta line, caption "operational, unofficial" |
| AOI polygon | `Geometry.geom` from the run | Dashed envelope |
| Burnable clip | LANDFIRE FBFM40 mask | Light hatch; urban/water (91/98) not hatched |
| Arrival field | `spread_run` raster | Choropleth of `arrival_hours` 0-72, NaN transparent |
| P(burn 72) | ensemble fraction | Optional second choropleth, mutually exclusive with arrival |
| Wind | HRRR u/v | Single barb at AOI centroid, not a streamfield |

Color scale for arrival: 0 h dark red → 72 h pale yellow → unreached transparent.
Legend required. Units hours.

Coarse grid: if flag `aoi_coarse_advisory`, caption "AOI coarsened to the cell cap;
advisory, not a fine-scale perimeter."

Isotropic buffer: if `aoi_isotropic_buffer`, caption "No usable wind; buffer is
isotropic."

Empty AOI: `aoi_empty` → do not draw a fake blob.

Do not upload the GeoTIFF to the browser as a raw float grid if it is huge. Serve a
pre-rendered PNG or quantized tiles from the spread field already cached per
`incident_id`. The HTTP service already returns JSON arrays; a thin tile endpoint is
a UI-backend task, not a change to `model_infer`.

## 6. Ask flow delta

Tool order shown in the progress list must match FR-30:

`geocode?` → `nws` → `firms + wfigs + hrrr` → `pack E` → `mireye W` → **`spread_run`**
→ `model_infer` → `policy`

`spread_run` may take longer than V1 (LANDFIRE job + ensemble). NFR-1's 30 s ask
target can miss when LFPS queues. Show queue position if the LANDFIRE status payload
has `queuePosition`. Do not time out into a fake ETA.

## 7. Incident page (new)

URL: `/incidents/:irwin_id`

- Title: WFIGS incident name, acres/containment (null-safe)
- Shared `spread_field_version`, engine name, `n_members` (7 default)
- Map from 5.3
- Table of watched sites: in-AOI, ETA, action, last card link
- Copy: "One field per incident; nearby sites reuse it. Cost scales with active
  incidents, not site count."

No edit controls for the raster. Operators do not "tweak ROS" in the UI.

## 8. Response Dossier delta

### 8.1 AOI water map (FR-56)

Below the V1 ranked water list, a map + table of sources that have `lat`/`lng`
(grid cells). Site-point sources keep null coordinates and stay in the list only.

```
┌──────── map ────────┐  Lake Test          lake_reservoir
│  AOI  · water pts   │  34.004, -118.012   availability unknown
│  site pin           │  distance unknown   (Mireye has no waterbody distance)
└─────────────────────┘
```

Cap: `aoi.water_grid_max_cells` (25). If the agent returned fewer, do not interpolate
extra dots. De-dupe is already done by (type, name).

Section title: "Water inside the fire risk zone (AOI grid)". Caption: "Mireye fields
on a dense grid. USGS discharge is queried only at the site gage, not at every cell."

### 8.2 Fire-station routing (FR-57)

`FireStation.eta_source`:

| Value | Badge | Map |
|---|---|---|
| `osm_network` | "OSM road network" | Draw the route polyline if the API later exposes it; until then, station pin + site pin + ETA minutes from the card |
| `distance_proxy` | "Distance / speed estimate (V1 fallback)" | No fake road |
| null | no badge | Same as V1 (unknown ETA) |

Do not label a proxy ETA as "Google Maps" or "live traffic". There is no traffic API.

Show `eta_minutes_estimate` with one decimal only if the backend emitted a float;
do not round in a way that disagrees with the card (brief validator will fight you).

## 9. Flags (V2 additions)

| Flag | Label |
|---|---|
| `aoi_coarse_advisory` | AOI coarsened (advisory) |
| `aoi_isotropic_buffer` | AOI buffer isotropic (no wind) |
| `aoi_empty` | AOI empty |
| plus all V1 flags | unchanged |

## 10. Policy chrome

A read-only "why this action" drawer can show the versioned table from
`config/policy.yaml` (do not let the UI edit production thresholds without bumping
`version`):

- Distance: monitor > 15 km, prepare 4-10 km, protect < 4 km
- y_hat: prepare 0.4, protect 0.7
- ETA: evacuate window < 6 h, prepare 24-72 h
- Sigma suppress: y_hat sigma 0.35, eta_sigma 12 h
- Housing density evacuate 500 / km² and limited egress

This is documentation UX, not a hidden second policy.

## 11. Copy deck (V2)

Allowed:

- "Arrival estimate with sigma"
- "P(burn by 24 / 48 / 72 h) from the ensemble"
- "Raw spread field" for `baseline_y` when it is p_burn_72
- "Operational perimeter (unofficial)"
- "OSM road-network ETA" vs "distance/speed estimate"

Forbidden:

- "The fire will reach you at 4:12pm"
- "Official perimeter from our model"
- "ELMFIRE says X" unless `spread_field_version` actually starts with a real elmfire
  binary version. Today's default engine id is `rothermel_huygens_v1`. Show that string
  verbatim.
- Filling ETA with distance/speed of the fire front in the browser

## 12. States (added)

| State | UI |
|---|---|
| LANDFIRE queue | Ask progress: "LANDFIRE job queued (position N)" |
| spread_run 200 | Stamp `spread_field_version` even if site ETA is null |
| LANDFIRE / spread failure | `degraded`; V1 layout; no fake raster |
| Site outside AOI | Pin on incident map, grey; ETA column "outside field" |
| Untrained calibration head | Banner: H2 not evaluable / model untrained; raw field still shown |
| H2 unevaluable | Audit page may link `scripts/evaluate_h2.py` output; do not display a fake "H2 passed" badge |
| Water grid skipped (no AOI) | V1 water list only |
| OSM empty | `eta_source=distance_proxy` or null; no route line |

## 13. Non-goals for V2 UI

- Learned-from-scratch spread viewer (SRS rejects this).
- Cell2Fire GPL surface.
- Live EO / smoke chips on the hot path.
- Editing ensemble members by hand.
- Public-facing "will my house burn?" map.

## 14. Implementation notes

- Reuse V1 components (`ActionPill`, `NullNumber`, `FlagChip`, `Brief`).
- New components: `EtaClock`, `PBurnBars`, `SpreadMap`, `WaterMap`, `EtaSourceBadge`.
- Card JSON: dump with `by_alias=True` so `p_burn_by_T` keys are `"24"|"48"|"72"` as
  in SRS 4.4 (the ask CLI already does this).
- Suggested extra read APIs: `GET /incidents/:irwin_id/field` (arrival PNG + extent +
  `spread_field_version`), `GET /incidents/:irwin_id/aoi` (GeoJSON),
  `GET /cards/:id/water-map` (sources with lat/lng).
- Keep Slack as push; extend Slack blocks later to include ETA and
  `spread_field_version`. The web map is the place those fields earn their keep.
- Accessibility: ETA clock needs a text equivalent ("14 hours plus or minus 3 hours,
  arrival estimate with sigma"), not color-only bars.
