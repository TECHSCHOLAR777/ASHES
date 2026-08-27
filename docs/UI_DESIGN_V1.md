# ASHES V1 UI design

Design for a site-ops UI on top of the V1 wildfire copilot. The backend already exists:
watch loop, ask CLI, Slack/email delivery, ActionCard, ResponseCard. This document
specifies the first web UI. It does not invent new scores or a learned map.

The product is a decision tool for a known book of commercial and industrial sites, not a
public-safety broadcaster. It never downgrades an NWS warning and never substitutes for an
evacuation order.

## 1. Users and jobs

| User | Job |
|---|---|
| Site ops / facilities | See whether any watched site needs prepare / protect / evacuate, and why. |
| Duty manager | Ask about one lat/lng during an unfolding week and get a cited card in ~30 s. |
| Response lead | When the card is `protect_asset` or `evacuate_site`, open the Response Dossier (water, access, hazmat, agency). |
| Auditor | Reconstruct every number from citations, W sources, and the daily JSONL log. |

Non-users: the general public, insurance pricing, InciWeb replacement.

## 2. Design principles (non-negotiable)

1. **Cards are the UI.** Every screen is a view of an ActionCard and, when escalated, its
   ResponseCard. Do not compute a second score in the browser.
2. **Null means unknown.** `acres`, `containment_pct`, water `distance_m`, fire-station ETA,
   and similar fields render as "n/a" or "unknown". Never coerce to 0.
3. **The LLM only writes the brief.** The brief is copy-only prose already validated against
   the card. If validation failed, show the fallback brief and a "brief replaced" chip.
4. **Sigma is always visible** next to `y_hat`. A probability without uncertainty is a bug.
5. **Citations are one click away.** Every number on a card has a source URL or a W-source
   vintage. Uncited numbers are not shown.
6. **CAP is never suppressed.** A Red Flag (or other CAP) on the card stays in the weather
   strip even when `y_hat` is low.

These rules match SRS 3.1, 4.3-4.5, FR-34 through FR-55, and NFR-17.

## 3. Information architecture

```
App
├── Watch board          /                    book of sites, last card per site
├── Site                 /sites/:site_id      timeline of cards for one site
├── Card                 /cards/:card_id      ActionCard (+ ResponseCard if any)
├── Ask                  /ask                 one-shot lat/lng query
├── Sites admin          /admin/sites         YAML-equivalent editor
└── Audit                /audit               tool log + credit spend (read-only)
```

Auth is out of scope for V1 UI (internal tool behind existing org SSO). Do not add a
public "am I in danger?" search.

## 4. Visual language

Action colors (already used in Slack delivery):

| Action | Hex | Use |
|---|---|---|
| `no_action` | `#808080` | Quiet / grey |
| `monitor` | `#2E86C1` | Info blue |
| `prepare` | `#F1C40F` | Warning yellow |
| `protect_asset` | `#E67E22` | Orange |
| `evacuate_site` | `#C0392B` | Red, highest visual weight |
| `inspect_after` | `#8E44AD` | Purple, post-incident |

Typography: system UI sans. Numeric values in tabular figures. Distances shown as km when
>= 1000 m (`113.3 km`), otherwise meters. Containment as percent if present (`10%`), else
"n/a".

Map: one basemap, site pin, optional WFIGS perimeter polyline. V1 does **not** draw a
spread raster. The wind-projected ROS ellipse is an E-side feature, not a perimeter; if
shown, label it "72 h wind ellipse (proxy), not an official perimeter".

## 5. Screens

### 5.1 Watch board

Primary screen. One row per site in `config/sites.yaml`.

```
┌─────────────────────────────────────────────────────────────────┐
│ ASHES V1          Watch                    [Ask]  [Sites]       │
│ Last poll 14:02 UTC · next in 3 min · 5 sites                   │
├──────────────┬──────────┬────────────┬──────────┬───────────────┤
│ Site         │ Action   │ Distance   │ y_hat    │ Weather       │
├──────────────┼──────────┼────────────┼──────────┼───────────────┤
│ Riverside    │ MONITOR  │ 12.4 km    │ 0.22±0.18│ RF no · W 4m/s│
│ Warehouse    │          │ LAC-301937 │ vs 0.11  │               │
├──────────────┼──────────┼────────────┼──────────┼───────────────┤
│ Napa Valley  │ NO ACTION│ n/a        │ 0.10±0.50│ RF no         │
│ Winery       │          │ no incident│          │               │
└──────────────┴──────────┴────────────┴──────────┴───────────────┘
 Flags strip: firms_unavailable  stale_E
```

Row fields (from last ActionCard, or empty state if never polled):

- Site name, `site_id`
- `action` pill in the color table
- `incident.incident_name` + `dist_perimeter_m` (null → "n/a")
- `y_hat` with `sigma` as ±, and `baseline_y` as "vs baseline"
- Red Flag boolean, wind speed / RH if present
- Flag chips: `degraded`, `firms_unavailable`, `stale_E`, `perimeter_unofficial`,
  `fhsz_missing`, `no_ros_high_sigma`

Sort: escalate first (`evacuate_site` > `protect_asset` > `prepare` > `inspect_after` >
`monitor` > `no_action`), then nearest perimeter.

Click row → Site timeline. Click action pill → that Card.

Empty watch: "No cards yet. The 5-10 min watch loop has not completed a poll, or ask from
the Ask page."

Dedup: if the backend suppressed a repeat card inside the 30 min window, do not flash a
new row. Show "unchanged since 13:40 UTC".

### 5.2 Ask

Single-purpose form. Matches `scripts/ask.py`.

- Lat, lng (required). Optional name. Optional question text (stored for the log; the
  model does not answer the question).
- Submit → blocking progress list of tools in FR-30 order:
  `nws_alerts` → `firms + wfigs + hrrr` → `pack E` → `mireye W` → `model_infer` → `policy`
- Target: card on screen in 30 s p95 (NFR-1). If slower, keep the tool list ticking; do
  not invent a partial score.
- Result is the Card screen for the new `card_id`. If action is protect/evacuate, the
  Response Dossier is on the same page (ask mode runs the response agent synchronously).

Do not offer `/v1/ask` or a free-text "will it burn?" box. Mireye is field-fetch only.

### 5.3 Card (ActionCard)

One page, four bands.

**Band A — decision**

- Huge action label (`EVACUATE SITE`) in the action color
- Site name, lat/lng, generated_at
- Recommended actions as a checklist (from `recommended_actions`, not LLM)
- Policy reasons (`reasons`) as bullets
- Flag chips

**Band B — scores**

| Label | Field | Copy |
|---|---|---|
| Model P(in perimeter 72 h) | `y_hat` | "Model score" |
| Uncertainty | `sigma` | always shown |
| Distance + ROS baseline | `baseline_y` | "Baseline the model must beat" |
| Policy | `policy_version` | e.g. `v1.0.0` |
| Model | `model_version` | e.g. `h_fire_v0_untrained` |

V1 `eta_hours`, `p_burn_by_T`, `spread_field_version` are null. Hide the ETA block entirely.
Do not show "ETA: n/a" as if a spread engine ran.

**Band C — situation**

- Incident: name, IRWIN id, acres, containment, hours since discovery. Null acres stay
  "n/a" (FR-40).
- Distance to perimeter (geodesic, code-computed)
- Weather: Red Flag, wind, RH, HRRR valid time
- Mini-map: site pin + WFIGS perimeter if rings exist. Caption: "Operational perimeter
  (unofficial)" when `perimeter_unofficial` is set.

**Band D — brief and provenance**

- Brief (validated prose). Link "why this wording" to the validator result if it was
  replaced.
- Citations list (source, URL, fetched_at, field)
- W sources (field, source_url, vintage, confidence)
- E product times (nws_cap, firms, wfigs, hrrr)

Footer: `card_id`, `policy_version`, `model_version`. JSON download of the card (same
schema as `ActionCard.model_dump`).

### 5.4 Response Dossier

Only mounted when parent action is `protect_asset` or `evacuate_site`. Same thread
metaphor as Slack: it is a child of the ActionCard (`triggered_by_action_card`).

Sections, in this order:

1. **Water** — ranked `water_sources`. Type, name, availability, discharge_cfs, drought
   note. If `distance_m` is null (Mireye has no flowline/waterbody distance), show
   "distance unknown", never a guess.
2. **USGS gage** — `usgs_gage_summary` (name, cfs, class, fetched_at).
3. **Access** — `access_routes` ranked by road class / surface / usability.
4. **Fire station** — name, distance, ETA minutes. Caption: "Distance / speed estimate,
   not a routing API" (FR-48).
5. **Airport** — staging field, distance only, no flight time.
6. **Hazmat** — priority critical / high / medium. Tiers: < 500 m critical, < 2 km high,
   < 5 km medium. Beyond 5 km omitted by the agent; do not add them in UI.
7. **Environmental constraints** — retardant restricted / coordinate first.
8. **Agency, comms, evacuation load, structure** — as on the ResponseCard.
9. **Brief** — copy-only response brief.
10. **Citations / W sources**

Empty lists: "none within radius" / "none within 5 km", matching Slack copy. Do not
fabricate a municipal hydrant.

### 5.5 Site timeline

Chronological ActionCards for one `site_id`. State store already keeps last action and
last card. Show action transitions (`monitor` → `prepare` → `protect_asset`) and
`inspect_after` when containment hits 100% and the site was previously in play.

### 5.6 Sites admin

Form equivalent of `config/sites.yaml`: `site_id`, name, lat/lng or address, slack
channel, email. Saving writes the YAML the watch loop already reads. No extra database
in V1 unless ops insists; file-backed is enough for a book of ~10 sites.

### 5.7 Audit

Read-only. Tail `data/logs/YYYY-MM-DD.jsonl` grouped by `site_id` / `card_id`. Show
tool name, latency, error. Surface `scripts/credit_report.py` spend vs the V1 credit
envelope. Operators use this when a number on a card is challenged.

## 6. Component rules

### Action pill

Closed enum only. Invalid action is a backend bug; do not map it to "alert".

### NullNumber

Shared component: if value is `null`/`undefined`, render the string "n/a" in muted
type. Never `0`, never "—", never a sparkline of zeros.

### FlagChip

Human labels (do not invent extra flags):

| Flag | Label |
|---|---|
| `perimeter_unofficial` | Unofficial perimeter |
| `no_ros_high_sigma` | Evacuate suppressed (high sigma) |
| `fhsz_missing` | FHSZ missing |
| `firms_only` | FIRMS only |
| `stale_E` | Stale weather/E |
| `firms_unavailable` | FIRMS unavailable |
| `degraded` | Degraded inputs |

### Brief

Render as paragraphs. Do not restyle numbers; they already copy the card. If the
validator replaced the LLM text, show a muted banner: "Brief failed validation and was
replaced with the fallback. Offending tokens logged."

## 7. States

| State | UI |
|---|---|
| First-run, no cards | Empty watch + CTA to Ask or wait for poll |
| Poll in progress | Subtle "polling 5 sites" on the board, no fake actions |
| Tool failure | Card still shown with `degraded` / `firms_unavailable` / `stale_E` |
| Untrained model | `model_version` = `h_fire_v0_untrained`; show a persistent banner that y_hat has no research skill |
| Slack/email unset | Delivery status "log-only" (see `config/README_MISSING.md`) |
| Protect/evacuate | Auto-open Response Dossier; do not make the user hunt for it |
| Ask timeout | Keep spinner; on hard fail, show which tool died from the log, no partial y_hat |

## 8. Copy deck (V1)

- Action is decided by the policy table, not the assistant.
- "Distance to perimeter" is geodesic to the operational WFIGS polygon, unofficial.
- "Model score" is P(site inside final perimeter within 72 h), calibrated, with sigma.
- "Baseline" is the distance + ROS ellipse, not a second model.
- Never say "the fire will arrive at HH:MM" in V1. There is no ETA field.

## 9. Non-goals for V1 UI

- Spread rasters, ETA clocks, P(burn by T) charts (V2).
- OSM routing map (V2).
- Dense AOI water grid (V2).
- Parcel owner lookup, `/v1/ask`, live EO chips.
- Dollar loss, debris flow, public alerting.

## 10. Implementation notes for a frontend

- Consume the existing Python agent; do not reimplement policy in JS.
- Suggested read API (not built yet): `GET /sites`, `GET /sites/:id/cards`,
  `GET /cards/:id`, `POST /ask {lat,lng,q}`, `GET /logs`. Bodies are the Pydantic
  schemas in `src/schemas/action_card.py` and `src/schemas/response_card.py`.
- Slack remains the primary push channel (FR-39). The web UI is the place to inspect
  and ask; it does not replace Slack for night-duty paging unless ops chooses that later.
- Accessibility: action must not be color-only (include the word EVACUATE / PROTECT).
  Target WCAG 2.2 AA for contrast on the red/orange pills.
