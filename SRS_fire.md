# Software Requirements Specification (SRS)

## Wildfire Site-Event Copilot

Document status: FINAL, end-to-end engineering deliverable
Peril scope: US wildfire (WUI + forest / rangeland fire affecting a named site)
Structure: IEEE-830 adapted for an ML + agent product
Ground-truth source docs: `plan_fire.md`, `plan_fire_architecture_qa.md`, `agent_1.md`, `mireye_into_model.md`

This SRS does not contradict the source docs; it consolidates and formalizes them into a buildable
specification. Every place where this SRS makes a NEW decision not already settled in the source
docs is marked inline with the tag **[NEW DECISION]** and is also collected in Section 10.7 for
review. The plan is delegated-engine + calibration for spread; the learned-spread-from-scratch
approach is explicitly rejected and is not resurrected anywhere in this document.

**Budget and quality posture (governs every credit/COGS decision below).** Mireye credits are
**abundant, not scarce** (>1,000,000 credits available). The build-program cap of 25,000
credits/person/month in `agent_1.md` is a *program* limit, not our constraint, and is NOT the design
driver here. Consequences, applied throughout this document:

- **V1 is quality-first within a ~300,000-400,000 credit budget.** V1 does NOT ship a minimal
  11-field allow-list; it ships a curated, fire-relevant **~60-field Mireye join** (2.2.2 / 6.3) plus
  optional parcel-geometry precision, because Mireye's own thesis is that value = the join of many
  layers no one has fused (`agent_1.md`).
- **V2 spends as many credits as quality needs** (dense AOI grid enrichment, richer refresh).
- **W caching exists for freshness, latency, and reproducibility - NOT to survive a credit cap.** We
  still `quote` before `fetch` (free, avoids waste and pins reproducibility), and we still avoid
  genuinely wasteful calls (e.g. `/v1/ask` for facts, refetching static terrain), but we never trade
  decision quality to save credits.
- Discipline that remains for engineering reasons (not cost): typed/masked encoding, vintage
  tracking, batch <=25, 60 rpm rate limit, and never fabricating a missing value.

---

## Table of Contents

1. Introduction
2. Overall Description
3. System Architecture (end-to-end)
4. External Interface Requirements
5. Functional Requirements
6. Data and Model Requirements
7. Non-Functional Requirements
8. V1 vs V2 (end-to-end)
9. Optional / Stretch
10. Risks, Open Questions, and UNVERIFIED Items
11. Milestones / Phased Roadmap
12. Acceptance Criteria

---

## 1. Introduction

### 1.1 Purpose

This document specifies the complete requirements for a **US site-level wildfire event copilot**:
a system that, while a wildfire is live or imminent, joins live fire signals, cited site physics,
and a trained model to emit a **closed, cited action** that a human takes in the next hours for a
**book of named sites** (plants, warehouses, campuses, timber yards).

The SRS is the engineering contract for V1 (MVP) and V2, with V3+ explicitly marked optional. It is
written for the engineers, ML researchers, and product owner who will build, evaluate, and operate
the system.

### 1.2 Product Scope

**In scope.** For one or many US coordinates, during an active or imminent fire, the product:

- fetches **live E** (fire weather alerts, thermal hotspots, incident metadata, operational
  perimeters, and gridded weather) that Mireye does not serve;
- fetches **cited W** (fuels, terrain, hazard-zone, burn history) from the Mireye API;
- computes a **spread-derived signal** (V1: a wind-projected rate-of-spread ellipse feature;
  V2: a delegated operational spread field producing per-site arrival time);
- runs a **trained model** (`h_fire`) that produces a calibrated site-level score with uncertainty;
- runs an **agent** that orchestrates the tools deterministically and emits an **ActionCard** with
  citations and a recommended action.

**Out of scope (explicitly not this product).**

- Replacing InciWeb / FIRMS maps / Google SOS (consumer "am I in danger").
- Replacing CAL FIRE FHSZ or FEMA NRI wildfire frequency (Mireye already serves those as W).
- Publishing tomorrow's official perimeter as operational truth.
- A national nowcast of ignition.
- Post-fire debris-flow / landslide as a modeled head (alert-only if CAP mentions it; no `h_debris`).
- Parcel-owner lookup in the hot loop (300-credit Mireye parcel fields are excluded).

### 1.3 The One-Line Product

**An unattended, cited fire-week ActionCard for a book of US sites: it watches live fire signals,
enriches each site with cited physics, estimates when/whether the fire reaches the site, and
recommends monitor / prepare / protect / evacuate without inventing numbers.**

### 1.4 Intended Users and the Decision They Make

| User class | Who they are | The decision the product supports |
|---|---|---|
| Primary buyer / operator | Facilities or risk lead responsible for >=20 US sites in fire-exposed geographies, who already toggles FIRMS/InciWeb during fire week | Per-site, per-shift: monitor, pre-stage crews/assets, shut down, or evacuate a specific site in the next hours |
| Secondary consumer | Site/plant manager receiving the alert | Execute the recommended action for their one site |
| Internal ML/research user | Modeler evaluating skill and calibration | Decide whether W-conditioning earns its keep vs baselines (the research claim) |
| Internal ops user | On-call engineer running the watch loop and (V2) the spread engine | Keep the loop and simulations healthy; audit provenance |

The product is a **decision tool for a known book of sites**, not a public safety broadcaster. It
never downgrades an official NWS warning and never substitutes for evacuation orders.

### 1.5 Definitions and Acronyms

| Term | Definition |
|---|---|
| **E** | Complementary/realtime fetch component: live fire signals Mireye does not have (CAP, FIRMS, WFIGS, HRRR). Also used to mean the packed E feature block. |
| **W** | Mireye API component: cited site physics (fuels, terrain, hazard zone, burn history). Also the packed W feature block. Not a prompt; not live weather. |
| **X** | Optional EO chip/embedding (pre-event fuel texture). Demoted to a cached W-side vintage feature; absent from the V1 hot path. |
| **h_fire / theta** | The trained deep-learning model with genuine learned parameters (`theta`). The LLM is not the model. |
| **y / y_hat** | Ground-truth label / model prediction. V1 `y` = site inside final perimeter within 72 h of t0; V2 adds arrival time. |
| **sigma** | Predictive uncertainty attached to every score/ETA. Never omitted. |
| **AOI** | Area of interest geometry. V1 = point/parcel. V2 = incident-first buffer around an active perimeter. |
| **ActionCard** | The closed, cited output object (schema in 4.4). |
| **arrival time / ETA** | Estimated hours until the fire front reaches a site (V2 primary; feeds `eta_hours`). |
| **ROS** | Rate of spread of a fire front. V1 uses a wind-projected elliptical ROS as a feature. |
| **FIRMS** | NASA Fire Information for Resource Management System: MODIS/VIIRS/Landsat/GOES active-fire thermal detections. |
| **VIIRS / MODIS / GOES** | Thermal sensors delivered via FIRMS. VIIRS 375 m; MODIS 1 km; GOES sub-hourly ~2 km. |
| **WFIGS** | Wildland Fire Interagency Geospatial Services: current incident locations (IRWIN) and operational perimeters. |
| **IRWIN** | Integrated Reporting of Wildland-Fire Information: incident id, name, acres, containment, discovery. |
| **NWS / CAP** | National Weather Service / Common Alerting Protocol: Red Flag / fire-weather alerts by point. |
| **SPC** | NOAA Storm Prediction Center fire-weather outlook (1-8 day). |
| **HRRR** | High-Resolution Rapid Refresh: NOAA 3 km hourly weather model (10 m wind U/V, temp, RH). Free via NODD; accessed via `Herbie`. |
| **Herbie** | Python library for fetching HRRR and other NWP from NOAA Open Data Dissemination (NODD). |
| **LANDFIRE** | USGS/USFS gridded fuel/terrain: FBFM40 (Scott & Burgan) / FBFM13 fuel models, canopy (CBD/CBH/CC/CH), DEM. Required by the spread engine (V2). |
| **FBFM40 / FBFM13** | Fire Behavior Fuel Models (Scott & Burgan 40 / Anderson 13) used by physics spread engines. |
| **ELMFIRE** | Eulerian Level-set Model of FIRE spread; operational open-source engine, **EPL-2.0**; powers Pyrecast; Cloudfire microservices auto-pull LANDFIRE + weather + ignition. Delegated engine of choice (V2). |
| **Cell2Fire** | Cellular-automata spread engine, **GPL-3.0**; alternative to ELMFIRE. Copyleft caution for closed SaaS. |
| **Cloudfire** | ELMFIRE's data-fetch microservices (`fuel_wx_ign.py`) that pull LANDFIRE fuel + weather + ignition. |
| **Pyrecast** | Operational forecasting service built on ELMFIRE (evidence ELMFIRE is production-grade). |
| **MTBS** | Monitoring Trends in Burn Severity: gold burned-area boundaries (USGS/USFS), public domain, ~1 yr lag, large fires only. Training label source. |
| **WSTS / WSTS+** | WildfireSpreadTS (NeurIPS 2023) and its WACV 2026 extension: next-day spread benchmark. SOTA AP ~0.47; cited as the reason to delegate, not learn, spread dynamics. |
| **FiLM** | Feature-wise Linear Modulation: `MLP(W) -> (gamma, beta)` applied to vision/E features. |
| **AlphaEarth / Prithvi-EO / Clay** | EO foundation encoders/embeddings (V3 EO on-ramps). AlphaEarth = 64-D annual embedding (zero-GPU); Prithvi-EO-2.0 = fine-tunable HLS/SWIR encoder. |
| **HUC / HUC-6 / HUC-12** | Hydrologic Unit Code watershed boundaries, used to define held-out geographic splits. |
| **RAWS / MesoWest** | Remote Automated Weather Stations / MesoWest access. Point wind/RH augmentation; **license UNVERIFIED**; optional. |
| **AP / PR-AUC / Brier** | Average Precision / Precision-Recall AUC / Brier score: primary skill and calibration metrics. |
| **COGS** | Cost of goods sold. Mireye credits, spread-engine compute, and OpenAI tokens are COGS. |

### 1.6 References

- `plan_fire.md` - end-to-end fire plan (four components E/W/DL/Agent, two dimensions research/product).
- `plan_fire_architecture_qa.md` - first-principles architecture audit and the "Spread as the spine" section (most current architecture; delegated engine + W-calibration; incident-first geometry; HRRR + LANDFIRE feeds; phased P0->P3).
- `agent_1.md` - Mireye API real-world knowledge (327 fields, presets, costs, credits, limits).
- `mireye_into_model.md` - how Mireye fields become the model feature vector (typed encoding, concat/FiLM/cross-attn, requirement for labels y).
- Supporting (context only, not depended on): `disaster_four_component_plan.md`, `honest_framing.md`, `final_clarity_updated.md`.
- External (verified in the architecture QA): ELMFIRE (EPL-2.0), Cell2Fire (GPL-3.0), LANDFIRE, HRRR/Herbie, MTBS, WSTS+, SILVIS WUI, Prithvi-EO-2.0, AlphaEarth.

### 1.7 Honest scope across wildfire management dimensions

This section records what the system genuinely does and does not do across the six recognized phases of
wildfire management. It is written as an anti-sycophancy gate: any future feature claim must map to a
row here and must upgrade the rating honestly.

| Dimension | Rating | What the system does | What it does NOT do |
|---|---|---|---|
| **MONITORING** | STRONG | 5-10 min watch loop across a book of sites; FIRMS VIIRS/MODIS/GOES thermal + FRP; WFIGS incidents + perimeters; NWS CAP Red Flag alerts; HRRR weather; geodesic distance-to-front; cited ActionCard on every state change; zero invented numbers | Consumer "am I in danger" alerting; national nowcast of ignition; replacing InciWeb or official SOS systems |
| **FORECASTING** | MODERATE | V1: 72 h site-in-perimeter probability with calibrated sigma; V2: per-site fire arrival time / P(burn by T{24,48,72}) from delegated ELMFIRE + Mireye-W calibration; genuine held-out skill claim with a falsifier | Ignition-risk forecasting; national fire-weather outlook; multi-week seasonal prediction; forecasting where a new fire starts |
| **RESPONSE SUPPORT** | MODERATE (V1) → STRONG (V2) | Mireye-W powered Response Dossier on `protect_asset`/`evacuate_site`: ranked water sources by distance/permanence/discharge (streams, lakes, municipal, wetlands, dams), access route quality, fire-station ETA estimate, hazmat priority list, comms availability, environmental retardant constraints, responsible agency; V2 adds AOI water map | Real-time truck routing (uses road-distance proxy, not live traffic); replacing NIFC/ICS-209 official incident management tools; coordinating actual crews; predicting retardant effectiveness |
| **IMPACT ANALYSIS** | MODERATE | WUI exposure at every site (building class/footprint/height, housing density, hazmat proximity); V2 ETA + sigma enables a probability-of-loss framing; compounding hazard fields (EPA RMP, RCRA, UST, transmission lines); responsible agency for liability | Formal damage-loss model with dollar estimates; debris-flow / erosion modeling; insurance loss-ratio computation; regional economic impact |
| **PREVENTION** | WEAK | Static hazard characterization from W (wildfire_annual_frequency, FHSZ class, burn history, lightning density) can power a pre-season site-risk report; identifying high-risk sites in a portfolio for pre-fire investment decisions | Active prevention recommendations (fuel-break siting, prescribed burn scheduling, vegetation management); policy or code enforcement tools; community-level risk reduction programs |
| **RECOVERY** | VERY WEAK | `inspect_after` action and `most_recent_burn_year`; alert passthrough if CAP mentions debris flow | Post-fire debris-flow or erosion head (no labels, no model); damage assessment; ecosystem recovery timeline; insurance claim support; MTBS label is for training only, not a live damage product |

**Reading the table:** the product is a real-time monitoring + near-term forecasting + response-support
tool for a defined book of commercial/industrial sites during an active fire. It is not an all-phases
disaster OS. Prevention and Recovery are weak and should not be over-sold. Response Support becomes
strong only when the Response Agent (see 2.2.5 / 5.12) is built. Impact Analysis is genuine but
scoped to site-level exposure, not regional loss. Claim only what the table says.

---

## 2. Overall Description

### 2.1 Product Perspective

The product is a **self-contained orchestration + ML system** layered on top of public data sources
and one commercial API (Mireye). It is not a GIS platform, not a consumer alert app, and not a
replacement for any official feed. Its defensible core is the **join**: fusing live fire geometry
(E), cited site physics (W), and a delegated spread signal into a **trained, calibrated** site score
that a deterministic policy converts into an action. The LLM is the orchestrator and briefer, never
the predictor.

### 2.2 The Four Components

**2.2.1 E - Complementary / realtime fetch.** Mireye has no live perimeter, no VIIRS hotspot stream,
no IRWIN incident, and no live weather. E supplies exactly these gaps:

- NWS CAP fire-weather / Red Flag alerts (point).
- SPC fire-weather outlook (regional, 1-8 day).
- NASA FIRMS active-fire thermal (VIIRS 375 m NOAA-20/21, MODIS, GOES) - the live "imagery" that
  sees through smoke.
- WFIGS incident locations (IRWIN) and operational perimeters.
- **HRRR gridded weather** (10 m wind U/V, temp, RH) - required for ROS and spread; added at V1.

E is refreshed every 5-10 minutes in the watch loop. Distances (site -> perimeter/front) are
computed geodesically in code, never by the LLM.

**2.2.2 W - Mireye API.** Cited site physics fetched via `geocode -> quote -> fetch/batch`, never via
`/v1/ask`, never parcel-owner in the hot loop. Because credits are abundant (see posture above) and
Mireye's value is the **join**, V1 fetches a curated **Fire Intelligence Set** (~60 fire-relevant
fields across nine roles), not a minimal allow-list. The set is chosen for fire signal, not
cardinality - siting-only fields (grid queue, solar/gas/interconnect economics) are excluded because
they carry no fire information. Roles (full field list in 6.3):

```
A. Fuel & vegetation      lcms_class, land_use_class, tree_canopy_pct, ndvi_current,
                          ndvi_change_5y, cdl_class, is_cultivated, dominant_crop_5y
B. Terrain (spread)       elevation, slope_degrees, aspect_degrees, aspect_cardinal,
                          soil_drainage_class, surface_water_permanence_pct,
                          soil_available_water_capacity
C. Fire-weather climo     mean_wind_speed_100m_ms, prevailing_wind_direction_100m_cardinal,
   (VINTAGE, not live)    weibull_k_100m, near_surface_wind_speed_annual_mean_ms,
                          mean_annual_dry_bulb_temperature_degc, mean_annual_relative_humidity_pct,
                          days_above_32c_annual_count, drought_category,
                          mean_annual_snow_cover_days, aerosol_optical_depth_annual_mean
D. Fire hazard priors     wildfire_annual_frequency, lightning_annual_flash_days,
                          fire_hazard_severity_zone_class, fire_hazard_responsibility_area,
                          nearest_fire_perimeter_distance_m, most_recent_burn_year,
                          landslide_susceptibility_index
E. Ignition sources       nearest_transmission_line_distance_m,
                          nearest_osm_transmission_line_distance_m, nearest_major_road_distance_m
F. Exposure / value       primary_building_height_m, primary_building_footprint_sqm,
                          primary_building_num_floors, primary_building_overture_class,
                          housing_units_within_1km, housing_units_density_per_km2, poi_count_1km
G. Access / egress        nearest_fire_station_distance_m, nearest_fire_station_name,
   / response             nearest_hospital_distance_m, nearest_road_distance_m, nearest_road_class,
                          nearest_road_surface, nearest_major_road_name, nearest_major_road_class,
                          roads_within_500m_count, total_road_length_within_500m_m,
                          nearest_school_distance_m
H. Compounding hazard     nearest_hazardous_facility_distance_m, nearest_rcra_tsd_distance_m,
                          nearest_ust_facility_distance_m, ust_facilities_within_1km_count
I. Firefighting water     nearest_usgs_gage_daily_discharge_cfs, domestic_well_households_per_km2,
   / governance           within_water_service_area, surface_management_agency
J. Response resources     nearest_waterbody_name, nearest_flowline_name, intersects_nhd_area,
   (ResponseAgent-add)    wetlands_within_100m_count, wetland_acres, high_hazard_dams_within_10km,
                          nearest_airport_name, nearest_airport_distance_m,
                          mobile_5g_coverage_class, nearest_antenna_structure_distance_m,
                          nearest_antenna_structure_height_m,
                          intersects_critical_habitat, critical_habitat_species,
                          intersects_protected_area, protected_area_designation,
                          protected_area_manager,
                          nearest_wastewater_plant_distance_m, nearest_wastewater_plant_name,
                          within_sewer_service_area, nearest_osm_substation_distance_m
```

Cost is ~75-85 credits/site including role J (mostly one-time; the set is dominated by static
terrain/hazard/climo/response fields). **Optional V1 parcel-geometry precision [NEW DECISION]:** for each site, one parcel field
(`parcel_boundary_geojson` or `parcel_area_m2`, 300 credits each) MAY be fetched once to pin the true
footprint for the 250-500 m fuel buffer; affordable at this budget, excluded only if the partner
opts out. `parcel_owner` remains excluded for privacy, not cost.

W is cached by (lat, lng, field-set, catalog vintage). Static roles (B terrain, D/E hazard & ignition
geometry, G access, H/I) are cached long (release-vintage TTL); only vintage-dynamic roles
(`ndvi_current`, `ndvi_change_5y`, `drought_category`) are refreshed on a short TTL during fire
season. Caching is for freshness/latency/reproducibility, not credit survival. Mireye supplies the
**calibration and context medium**, not the simulation medium and not live weather; `ndvi_current`
is a vintage fuels feature, never "greenness this minute."

**2.2.3 DL model - h_fire (theta).** A genuine trained model with learned parameters. Two coherent
variants sharing one `model_infer` contract:

- **Variant A (V1):** MLP/logistic on `[g(W); E]` (E includes the wind-ROS ellipse feature).
  Target = `P(site in final perimeter within 72 h of t0)`. Ships first.
- **Variant B (V2):** a delegated operational engine (ELMFIRE) produces an incident-centered
  arrival field; `h_fire` learns a **Mireye-W-conditioned calibration** of that field. Target =
  per-site **arrival time / P(burn by T) + sigma**.

The model never learns fire propagation physics from scratch. EO imagery, if ever used, enters as a
**precomputed embedding** (V3), not a live encoder.

**2.2.4 Agent.** Three modes, same typed tools:

- **watch** (cron over a YAML/CSV book of sites) — the primary loop; calls tools in a fixed order,
  packs E, fetches W, calls `model_infer`, applies the deterministic policy, emits an ActionCard +
  LLM brief.
- **ask** (on-demand lat/lng + question) — same pipeline, single site, interactive.
- **respond** (see 2.2.5) — triggered automatically when the main agent emits `protect_asset` or
  `evacuate_site`; runs the specialized Response Support Agent for the same site in parallel.

The LLM orchestrates tool calls in all modes but may not invent acres, containment, distances, or
scores, and may not call `/v1/ask` or suppress a CAP alert.

**2.2.5 Response Support Agent (parallel, specialized).** A second agent that runs **parallel** to
the main prediction agent whenever a site reaches `protect_asset` or `evacuate_site` action. Its job
is tactical resource intelligence: given the fire is coming, what exists on the ground to fight it,
protect the site, and evacuate safely?

The Response Agent pulls from the same cached W (role J fields added specifically for this purpose),
adds one fresh live query — the USGS gage daily discharge to confirm whether nearby streams have water
right now — and runs deterministic code (no learned model) to produce a **ResponseCard**. Outputs:

- **Water sources ranked**: all streams (by live discharge_cfs), lakes/waterbodies, wetlands, municipal
  water service, wells, wastewater treatment plants, dams within radius — each with distance,
  permanence, live availability (if gage present), and a usability rating.
- **Access routes**: road name, class, surface, distance — the best truck-accessible approach routes to
  the site, ranked by class (highway > arterial > local, surface = paved preferred).
- **Fire station ETA estimate**: nearest station distance + road-network proxy → estimated arrival
  minutes (code, never LLM).
- **Nearest air tanker staging**: nearest airport name + distance (for aerial suppression logistics).
- **Hazmat priority list**: EPA RMP, RCRA TSD, UST facilities, petroleum/gas pipelines, and
  transmission lines within radius, ranked by proximity and type — what must be protected or cleared
  first.
- **Environmental response constraints**: critical habitat (retardant restricted zones) and protected
  area designation + manager (who to call for access authorization).
- **Responsible agency**: `surface_management_agency` (BLM, USFS, NPS, FWS, DOD, state, local) —
  who owns the land and who is the primary response authority.
- **Comms availability**: 5G coverage class + nearest antenna tower distance/height for incident
  command setup.
- **Evacuation load**: housing units within 1 km, nearest hospital, nearest school (evacuation staging
  sensitivity).

The Response Agent uses no trained model; every number comes from a Mireye field, a USGS live gage,
or code computation. The ResponseCard is delivered alongside the ActionCard in the same Slack/email
thread. It does not replace official ICS/ICS-209 tools; it is a cited, structured briefing sheet for
the site manager and facilities operator.

### 2.3 The Two Dimensions

**2.3.1 Research dimension.** The falsifiable question of whether cited W, inside the net, adds skill
beyond distance-to-fire and beyond raw physics.

- V1 hypothesis (H1): FiLM/concat on cited W improves 72 h site-in-perimeter `y_hat` versus E-only
  and X-only on event + HUC/state holdout.
- V2 hypothesis (H2, stronger, subsumes H1): **Mireye-fuel-conditioned calibration of an operational
  spread model (ELMFIRE) improves site-level arrival-time / burn-probability skill versus the raw
  operational model, on event/HUC/state held-out fires.** Baseline = raw ELMFIRE (a hard,
  operational bar), plus exposure/climatology.

**2.3.2 Product dimension.** An unattended fire-week ActionCard for a book of sites: watch loop +
Slack/email delivery + thin dossier. Priced per **site-month of watch**; Mireye credits and (V2)
spread compute are COGS. Product success is "N sites, days unattended, zero invented numbers, human
prefers the card to browser tabs" - never mIoU, never stars.

The two dimensions weave through one `model_infer` contract: research changes what lives behind the
contract; product depends only on the contract's stable inputs/outputs. If research fails (no skill
from W), the product still ships as honest orchestration, and the "baked-W" story is paused, not
faked.

### 2.4 User Classes and Characteristics

See 1.4. The primary operator is technical enough to maintain a YAML/CSV site list and read a
structured card, but is not a GIS analyst or an ML engineer. The card must be self-explanatory,
cited, and conservative (high sigma -> no overconfident evacuate).

### 2.5 Operating Environment

- Backend service (Linux) running the watch loop, tool clients, and `model_infer`.
- Model training offline (GPU optional; V1 tabular model trains on CPU/small GPU).
- V2 adds a **spread-engine service** (ELMFIRE, Fortran core, run as a microservice with a job queue)
  plus HRRR/LANDFIRE fetch clients.
- Delivery to Slack and/or email. Configuration via YAML/CSV.
- External dependencies over HTTPS: Mireye API/MCP, api.weather.gov (CAP), FIRMS area API, WFIGS
  services, NOAA NODD (HRRR via Herbie), LANDFIRE, OpenAI.

### 2.6 Design and Implementation Constraints

**2.6.1 Licensing (critical for a closed SaaS).**

| Dependency | License | Implication for closed SaaS |
|---|---|---|
| **ELMFIRE** | **EPL-2.0** (weak/file-level copyleft) | Recommended engine. Run as a **separate service over an API/queue** and keep any modifications to ELMFIRE source files disclosed per EPL; do not statically link ELMFIRE source into proprietary code. Calling it as an out-of-process service is the compliant integration boundary. |
| **Cell2Fire** | **GPL-3.0** (strong copyleft) | Alternative only. GPL-3.0 is safe to run as a fully separate, arms-length pipeline/process, but **risky to link into** the closed product; conveying a derivative could trigger source-disclosure of linked code. Prefer ELMFIRE; if Cell2Fire is used, isolate it behind a network/process boundary and get legal review. |
| Mireye API | Commercial | Rate/credit limits (2.6.2); `/v1/ask` banned for physical facts; parcel-owner (300 cr) excluded. |
| FIRMS, WFIGS, NWS/CAP, HRRR, LANDFIRE, MTBS | Public / open | Usable; honor User-Agent and quota etiquette. SNPP VIIRS delivery ends 2026-11-01 -> NOAA-20/21 only. |
| RAWS / MesoWest | **UNVERIFIED** | Do not depend on for V1/V2; optional V3 augmentation pending license verification. |

**Firm boundary [NEW DECISION - integration boundary formalized]:** the spread engine is always
invoked out-of-process behind a stable internal API (`spread_run`), so the copyleft engine never
links into the proprietary model/agent code. This is the licensing-compliant seam and is required.

**2.6.2 API rate limits and credits.**

- Mireye: 60 requests/minute; batch <=25 coordinates; ~1 credit/ordinary field/location;
  ~55-65 credits/site for the V1 Fire Intelligence Set; 300 credits per parcel field (optional
  footprint precision only; `parcel_owner` excluded for privacy). Credits are **abundant**
  (>1,000,000 available); the 25,000/person/month build-program cap is NOT our constraint. Always
  `quote` before `fetch` - for reproducibility and to avoid waste, not to survive a cap.
- FIRMS MAP_KEY numeric quota is UNVERIFIED -> use a token bucket and back off.
- OpenAI: token cost per brief; briefs are short and copy-only. OpenAI credits are also abundant;
  briefs are bounded for latency/consistency, not cost.
- HRRR/LANDFIRE: free; cost is bandwidth/compute, not per-call fees.

**2.6.3 Hard behavioral constraints.**

- Never publish an official perimeter as truth; always emit an **arrival estimate with sigma**.
- Never downgrade or suppress an NWS/CAP warning.
- Never invent acres, containment, distances, or scores; geodesic math in code, not LLM.
- Never use `/v1/ask` for physical facts; never fetch parcel-owner in the watch loop.
- Never learn fire-spread dynamics from scratch; delegate the engine.

### 2.7 Assumptions and Dependencies

- The book of sites is US-only coordinates (Mireye is US-only).
- Public feeds (FIRMS, WFIGS, CAP, HRRR, LANDFIRE, MTBS) remain available with current schemas.
- Mireye catalog vintages are stable enough to cache W for 24 h.
- Sufficient historical fires with MTBS/final-WFIGS perimeters and reconstructable E-as-of-t0 exist
  to train and, for V2, to calibrate against raw ELMFIRE on held-out fires.
- ELMFIRE + Cloudfire remain runnable and can auto-pull LANDFIRE + weather + ignition.

---

## 3. System Architecture (end-to-end)

### 3.1 Principle

E, W, and X meet **only** inside `model_infer` and the ActionCard - never inside the LLM. The LLM
resolves intent, calls typed tools, and writes a copy-only brief. All numbers originate from tools
or the model.

### 3.2 End-to-end data flow (textual diagram)

```
                         USER QUERY  ("is the plant at X at risk?" | watch cron over book of sites)
                                        |
                         [LLM intent + Mireye geocode]                     <- Mireye (point; no polygon)
                                        |
                         GEOMETRY RESOLUTION
                           V1: point / parcel  (book of sites spine)
                           V2: incident-first AOI = buffer around active WFIGS perimeter,
                               wind-projected, clipped to burnable fuel; then intersect book of sites
                                        |
        +-----------------------------------------------------------------------------------+
        | LIVE E  (per poll, 5-10 min)                                                       |
        |   NWS CAP alerts (point)                                                           |
        |   SPC outlook (regional)                                                           |
        |   FIRMS VIIRS/MODIS(/GOES) hotspots + FRP in radii (5/10/20 km)                    |
        |   WFIGS incidents (name, acres, containment, discovery) + perimeters (rasterized)  |
        |   HRRR wind U/V, temp, RH  (NEW at V1)                                             |
        |   -> pack E: rflag, firms_counts, frp, dist_perim_m, acres, containment,           |
        |      hours_since_discovery, perimeter_unofficial, wind_ros_ellipse (V1)            |
        +-----------------------------------------------------------------------------------+
                                        |
        +-----------------------------------------------------------------------------------+
        | MEDIUM (cached)                                                                    |
        |   Mireye Fire Intelligence Set ~60 fields (context/calib; long+short TTL) <- Mireye  |
        |   V2: LANDFIRE FBFM40 fuel + canopy (CBD/CBH/CC/CH) + DEM (raster) -> for engine   |
        |   V3 (opt): AlphaEarth 64-D / Prithvi embedding (vintage fuel feature)             |
        +-----------------------------------------------------------------------------------+
                                        |
        SPREAD SIGNAL
          V1 (Variant A): distance-to-front + wind-projected ROS ellipse   ==> a FEATURE in E
          V2 (Variant B): DELEGATED ENGINE (ELMFIRE via Cloudfire) per active incident
                          -> raw arrival-time / P(burn) FIELD over incident AOI
                                        |
        model_infer(site_id, X|null, W, E, spread, vintages)
          A: y_hat = P(in perimeter 72 h), sigma
          B: h_fire calibrates raw field with Mireye W -> calibrated ETA, P(burn<=T{24,48,72}), sigma
                                        |
        book of sites intersect field / point passthrough  ->  per-site risk + ETA
                                        |
        policy(y_hat | ETA, P, E)  ->  ActionCard (deterministic table)
                                        |
        LLM brief: copies ActionCard fields + citation URLs only (no new numbers)
                                        |
        DELIVERY: Slack / email;  LOG every tool JSON + model_version + spread_field_version
```

### 3.3 Integration boundaries and contracts

| Boundary | Producer -> Consumer | Contract (inputs -> outputs) |
|---|---|---|
| Geocode | LLM -> Mireye `geocode` | address/place -> {lat, lng, confidence, interpolation_flag} |
| E fetch | Agent -> CAP/FIRMS/WFIGS/HRRR clients | {lat,lng,bbox,radii,poll_time} -> raw feeds; then packed E vector + product_times |
| W fetch | Agent -> Mireye `quote`/`fetch(/batch)` | {lat,lng,field-set} -> typed cited records; cached by vintage |
| Spread (V2) | Agent -> `spread_run` (ELMFIRE service) | {incident_id, perimeter, HRRR window, LANDFIRE tile, ensemble_cfg} -> arrival-time/P(burn) field + spread_field_version |
| Model | Agent -> `model_infer` | {site_id, X\|null, W, E, spread, vintages} -> {y_hat, sigma, eta_hours?, p_burn_by_T?, model_version, baseline_y, spread_field_version?} |
| Policy | Agent -> policy table | {y_hat\|eta,p_burn,E flags} -> {action, reasons[]} (deterministic) |
| Output | Agent -> ActionCard schema | all of the above -> ActionCard (4.4) |
| Brief | Agent -> LLM | ActionCard + citation URLs -> prose brief that copies fields only |
| Delivery | Agent -> Slack/email | ActionCard + brief -> message |

The `model_infer` and ActionCard contracts are **stable across V1 and V2**. Behind `model_infer` the
implementation evolves (calibrated constant -> MLP -> FiLM -> delegated-field calibration); the agent
does not change.

---

## 4. External Interface Requirements

### 4.1 Data sources and APIs

| Source | Provides | Latency | Join key | Status |
|---|---|---|---|---|
| **Mireye** `api.mireye.com` (+ `/mcp`) | ~60-field Fire Intelligence Set, geocode, quote, batch fetch | seconds | lat/lng | VERIFIED (commercial; 60 rpm; ~55-65 cr/site; credits abundant) |
| **NWS CAP** `api.weather.gov/alerts/active?point=` | Red Flag / fire-weather alerts | seconds-min | lat/lng | VERIFIED (public; User-Agent + contact required) |
| **SPC fire-weather outlook** (RSS) | 1-8 day regional fire weather | hours | region | VERIFIED (public) |
| **NASA FIRMS** area API | VIIRS 375 m (NOAA-20/21), MODIS 1 km, GOES, Landsat thermal + FRP | US URT often <60 s | bbox | VERIFIED; **SNPP delivery ends 2026-11-01**; MAP_KEY quota UNVERIFIED |
| **WFIGS Incident Locations Current** | IRWIN: name, acres, containment, discovery, lat/lng | min-hours | IRWIN id, distance | VERIFIED (public; small fires drop off) |
| **WFIGS Interagency Perimeters Current (+ YTD)** | operational polygon | hours (IR cycle) | geodesic site->boundary | VERIFIED (public; not final MTBS) |
| **HRRR** via `Herbie`/NODD | 10 m wind U/V, temp, RH, 3 km hourly | hourly, minutes latency | grid cell | VERIFIED (free; add at V1) |
| **LANDFIRE** | FBFM40 fuel, canopy CBD/CBH/CC/CH, DEM (raster) | static (periodic releases) | tile | VERIFIED (free; add at V2 for the engine) |
| **MTBS** | final burned-area perimeters (labels) | ~1 yr lag; large fires | pixel/parcel | VERIFIED (public domain; training only) |
| **RAWS / MesoWest** | point wind/RH | minutes | station | **UNVERIFIED license**; optional V3 only |
| "**DeepMind FireCast v3 / firecast.deepmind.dev**" | (claimed spread API) | - | - | **UNVERIFIED / likely fabricated - DO NOT integrate or design against** |
| FARSITE/FlamMap/Technosylva | spread engines | - | - | Existence verified generally; 2026 automation/licensing UNVERIFIED; not the pick (ELMFIRE is) |

### 4.2 The agent / LLM interface (OpenAI)

- Provider: OpenAI (chat/completions with tool-calling). **[NEW DECISION: OpenAI named as the LLM
  provider for V1; a provider-agnostic wrapper is recommended so the model can be swapped.]**
- The LLM has access ONLY to the typed tools in 5.7; it cannot fabricate tool outputs.
- The LLM's write step receives the finalized ActionCard + citation URLs and must produce prose that
  copies those fields verbatim (numbers, action, citations). Any output introducing a number not in
  the ActionCard is rejected by a post-generation validator.
- The LLM may NOT: call `/v1/ask`; call parcel-owner; compute distances/scores; suppress or reword a
  CAP warning's severity.

### 4.3 Output interface

- **Slack** (primary) and **email** (secondary) delivery of the ActionCard + brief.
- All numeric content in the delivered message must be traceable to a citation in the card.

### 4.4 ActionCard schema

The ActionCard is the single closed output object. Fields marked (V2) are populated only in V2.

```
ActionCard {
  card_id: string                       # uuid
  generated_at: iso8601                 # product time, not feed time
  site: {
    site_id: string,
    name: string,
    lat: float, lng: float,
    mode: "point" | "parcel" | "aoi",
    geocode_confidence: string,
    range_interpolation: bool
  }
  action: "monitor" | "prepare" | "protect_asset" | "evacuate_site"
          | "inspect_after" | "no_action"       # closed enum
  eta_hours: float | null               # (V2 primary) hours to fire front; null in V1
  eta_sigma_hours: float | null         # (V2) uncertainty on ETA
  p_burn_by_T: { "24": float, "48": float, "72": float } | null   # (V2)
  y_hat: float | null                   # (V1 primary) P(in perimeter 72 h)
  sigma: float                          # predictive uncertainty (always present)
  baseline_y: float | null              # baseline number the model must beat (distance+ROS / raw ELMFIRE)
  incident: {
    irwin_id: string | null,
    incident_name: string | null,
    acres: float | null,                # copied from WFIGS; never invented
    containment_pct: float | null,
    dist_perimeter_m: float | null,     # geodesic, code-computed
    hours_since_discovery: float | null
  }
  weather: {
    red_flag: bool,
    wind_speed_ms: float | null,
    wind_dir_cardinal: string | null,
    rh_pct: float | null,
    hrrr_valid_time: iso8601 | null
  }
  recommended_actions: [ string ]       # human-readable steps derived from `action` + policy
  reasons: [ string ]                   # deterministic policy reasons that produced `action`
  flags: [ "perimeter_unofficial" | "no_ros_high_sigma" | "fhsz_missing"
         | "firms_only" | "stale_E" | "aoi_coarse_advisory" | ... ]
  citations: [ { source: string, url: string, fetched_at: iso8601, field?: string } ]
  e_product_times: [ { feed: string, product_time: iso8601 } ]
  w_sources: [ { field: string, source_url: string, vintage: string, confidence: string } ]
  model_version: string
  spread_field_version: string | null   # (V2)
  policy_version: string
}
```

Rules: `action` is always set by the deterministic policy, never the LLM. `sigma` is always present.
`acres`/`containment_pct` are copied from WFIGS or null; never fabricated. When ETA/`p_burn_by_T` are
present (V2), the card must also state they are "arrival estimate with sigma," never "the perimeter."

### 4.5 ResponseCard schema

Emitted by the Response Support Agent (2.2.5) when `action` >= `protect_asset`. Every number comes
from a Mireye field, a live USGS gage, or deterministic code — never from the LLM.

```
ResponseCard {
  card_id: string                       # uuid; links to parent ActionCard card_id
  triggered_by_action_card: string      # parent ActionCard card_id
  generated_at: iso8601
  site: { site_id, name, lat, lng }     # same as ActionCard

  # WATER SOURCES (ranked by usability for aerial/ground suppression)
  water_sources: [
    {
      type: "stream" | "lake_reservoir" | "wetland" | "municipal_water" |
            "well_field" | "wastewater_plant" | "dam_reservoir",
      name: string | null,
      distance_m: float,
      discharge_cfs: float | null,          # live USGS gage if available
      permanence_pct: float | null,         # surface_water_permanence_pct
      availability: "high" | "moderate" | "low" | "dry" | "unknown",
      note: string | null,                  # e.g. "drought D3 - expect low flow"
      source_url: string,
      fetched_at: iso8601
    }
  ]

  # ACCESS ROUTES (ranked by road class + surface)
  access_routes: [
    {
      road_name: string | null,
      road_class: string,                   # nearest_major_road_class / nearest_road_class
      surface: string,                      # paved / unpaved / unknown
      distance_m: float,
      usability: "excellent" | "good" | "limited" | "unknown"
    }
  ]

  # FIRE RESPONSE RESOURCES
  fire_station: {
    name: string | null,
    distance_m: float,
    eta_minutes_estimate: float | null      # distance / road proxy; never LLM
  }
  nearest_airport: {
    name: string | null,
    distance_m: float                       # air tanker / helicopter staging
  }

  # HAZMAT PRIORITY LIST (what must NOT be allowed to burn)
  hazmat_sites: [
    {
      type: "EPA_RMP" | "RCRA_TSD" | "UST" | "gas_pipeline" |
            "petroleum_pipeline" | "transmission_line" | "osm_substation",
      name: string | null,
      distance_m: float,
      priority: "critical" | "high" | "medium",
      note: string | null
    }
  ]

  # ENVIRONMENTAL RESPONSE CONSTRAINTS
  environmental_constraints: [
    {
      type: "critical_habitat" | "protected_area",
      species_or_designation: string | null,
      manager: string | null,
      constraint: "retardant_restricted" | "access_restricted" | "coordinate_first" | "normal"
    }
  ]

  # RESPONSIBLE AGENCY
  responsible_agency: string | null         # surface_management_agency (BLM/USFS/NPS/FWS/state/local)

  # COMMS (for incident command setup)
  comms: {
    mobile_5g_coverage: string | null,
    nearest_antenna_distance_m: float | null,
    nearest_antenna_height_m: float | null,
    fiber_available: bool | null
  }

  # EVACUATION LOAD
  evacuation: {
    housing_units_within_1km: float | null,
    housing_density_per_km2: float | null,
    nearest_hospital: { name: string | null, distance_m: float | null },
    nearest_school_distance_m: float | null   # evacuation staging sensitivity
  }

  # STRUCTURE TRIAGE
  structure: {
    height_m: float | null,
    footprint_sqm: float | null,
    overture_class: string | null             # house / commercial / industrial / etc.
  }

  # USGS LIVE STREAMFLOW SUMMARY
  usgs_gage_summary: {
    gage_name: string | null,
    distance_m: float | null,
    discharge_cfs: float | null,
    discharge_class: "high" | "normal" | "low" | "critically_low" | "unknown",
    fetched_at: iso8601 | null
  }

  citations: [ { source, url, fetched_at, field? } ]
  w_sources: [ { field, source_url, vintage, confidence } ]
  response_card_version: string
}
```

Rules: no trained model behind this card; no LLM-invented numbers; every distance is geodesic code;
`eta_minutes_estimate` is a road-distance proxy (distance/average truck speed), never a routing API
call (V2 may add real routing). Water-source availability is coded from `discharge_cfs` + `drought_category`
+ `surface_water_permanence_pct` by a deterministic function, not an LLM judgment.

---

## 5. Functional Requirements

Each FR is tagged **[V1]**, **[V2]**, or **[V3]**. "SHALL" is mandatory.

### 5.1 Query intake and geo-resolution

- **FR-1 [V1]** The system SHALL accept a book of sites as YAML/CSV (site_id, name, lat, lng or
  address) for watch mode, and a single lat/lng or address + question for ask mode.
- **FR-2 [V1]** For addresses, the system SHALL geocode via Mireye `geocode`, returning lat/lng and
  confidence, and SHALL flag `range_interpolation` when the score is treated as "this pad."
- **FR-3 [V1]** The system SHALL resolve geometry to `point` or `parcel` (optional 250-500 m
  footprint buffer for fuel context). It MAY fetch one parcel-geometry field
  (`parcel_boundary_geojson`/`parcel_area_m2`, 300 cr) to pin the true footprint. `parcel_owner`
  SHALL NOT be fetched (privacy, not cost).
- **FR-4 [V2]** The system SHALL support `aoi` geometry as an **incident-first** buffer around an
  active WFIGS perimeter (wind-projected, clipped to burnable fuel), then intersect the book of
  sites. AOI surfaces SHALL be flagged `aoi_coarse_advisory`.
- **FR-5 [V1]** The geometry contract SHALL be uniform: `{mode, geom, cells[], source_layer,
  vintage}` with `cells` length 1 for point/parcel; `model_infer` runs per cell.

### 5.2 Incident detection

- **FR-6 [V1]** For each site/poll, the system SHALL query WFIGS incidents and perimeters within a
  configured radius and compute geodesic site->perimeter distance in code.
- **FR-7 [V1]** The system SHALL query FIRMS hotspots in radii (5/10/20 km) and aggregate counts and
  FRP; it SHALL require WFIGS confirmation or multi-pass FIRMS persistence on non-developed LCMS
  before any action stronger than `monitor` (guards against flares/industry false positives).
- **FR-8 [V1]** When FIRMS shows fire but WFIGS has no incident, the system SHALL set
  `perimeter_unofficial` and cap the action at `monitor`.
- **FR-9 [V2]** The system SHALL maintain a registry of active incidents and run one spread field per
  incident, reused across all nearby sites.

### 5.3 E ingestion

- **FR-10 [V1]** The system SHALL fetch NWS CAP alerts by point and SHALL never suppress or downgrade
  a warning; CAP severity SHALL pass through to the card.
- **FR-11 [V1]** The system SHALL fetch SPC outlook for the region as a slow signal.
- **FR-12 [V1]** The system SHALL fetch HRRR wind U/V, temp, RH for each site/incident and record the
  HRRR valid time.
- **FR-13 [V1]** The system SHALL pack E into: `rflag`, FIRMS counts/FRP by radius, `dist_perim_m`,
  `acres`, `containment`, `hours_since_discovery`, `perimeter_unofficial`, and the V1 `wind_ros_ellipse`
  feature (5.5).
- **FR-14 [V1]** The system SHALL record a product time per feed and SHALL flag `stale_E` when any
  feed is older than the configured policy age.

### 5.4 W enrichment

- **FR-15 [V1]** The system SHALL `quote` then `fetch`/`fetch/batch` the ~60-field Fire Intelligence
  Set (2.2.2, roles A-I) from Mireye, never `/v1/ask`, batching <=25 coordinates. Quote precedes
  fetch for reproducibility and waste-avoidance, not rationing (credits abundant).
- **FR-16 [V1]** The system SHALL cache W by (lat, lng, field-set, catalog vintage) with a
  **role-dependent TTL** - long (release-vintage) for static roles (terrain, hazard/ignition
  geometry, access, water/governance), short (fire-season) for vintage-dynamic roles (NDVI, drought)
  - and SHALL pass `dataset_vintage`/`fetched_at` into the model. Caching is for
  freshness/latency/reproducibility, not credit rationing.
- **FR-17 [V1]** The system SHALL encode W as typed features with missing masks (continuous scaled,
  binary 0/1, ordered categorical as integer/embed, unordered categorical one-hot/embed); IDs
  (tract_geoid, huc) SHALL be join keys only, never features.
- **FR-18 [V1]** When a field is missing or low-confidence (e.g. FHSZ outside CA), the system SHALL
  mask it and SHALL NOT fabricate a value; it SHALL set `fhsz_missing` where relevant.

### 5.5 Spread field computation

- **FR-19 [V1]** The system SHALL compute a **wind-projected elliptical ROS feature** from HRRR wind
  plus a crude ROS, together with distance-to-front, and expose it as an E-side feature. This
  replaces any vague "crude ROS" with a concrete, versioned feature.
- **FR-20 [V2]** The system SHALL run a **delegated operational spread engine (ELMFIRE, EPL-2.0)** via
  Cloudfire, per active incident, out-of-process behind `spread_run`, producing a per-cell
  arrival-time / P(burn) field over the incident AOI with a `spread_field_version`.
- **FR-21 [V2]** The engine SHALL be fed LANDFIRE (FBFM40 fuel, canopy, DEM) + HRRR weather +
  WFIGS/FIRMS ignition/perimeter. Mireye `lcms_class`/`ndvi` SHALL NOT be fed as a Rothermel fuel
  model.
- **FR-22 [V2]** The system SHALL run an ensemble (wind/moisture perturbations) sufficient to produce
  a per-site sigma on ETA; ensemble size SHALL be configurable.
- **FR-23 [V3]** The system MAY run a distilled fast surrogate that emulates ELMFIRE for latency, only
  after V2 works.

### 5.6 Model inference and calibration

- **FR-24 [V1]** `model_infer` SHALL accept `{site_id, X|null, W, E, spread, vintages}` and return
  `{y_hat, sigma, model_version, baseline_y}`; V1 `y_hat` = P(in perimeter 72 h).
- **FR-25 [V1]** The model output SHALL be probability-calibrated (isotonic/Platt on held-out
  validation).
- **FR-26 [V2]** `model_infer` SHALL additionally return `{eta_hours, eta_sigma_hours, p_burn_by_T,
  spread_field_version}`; V2 `h_fire` SHALL be a **Mireye-W-conditioned calibration head** on the raw
  delegated arrival field, not a learner of propagation.
- **FR-27 [V1]** The system SHALL always attach `baseline_y` (V1: distance+ROS; V2: raw ELMFIRE) to
  enable the card and audits to show what the model beats.
- **FR-28 [V1]** The system SHALL never emit an official-perimeter claim; V2 outputs SHALL be labeled
  "arrival estimate with sigma."

### 5.7 Agent orchestration, tool-use, and memory

- **FR-29 [V1]** The agent SHALL expose exactly these typed tools, none LLM-inventable:
  `geocode`, `nws_alerts`, `spc_outlook`, `firms_bbox`, `wfigs_incidents`, `wfigs_perimeters`,
  `hrrr_weather`, `mireye_quote`, `mireye_fetch`, `dist_geodesic`, `model_infer`,
  and (V2) `spread_run`, (V3) `stac_chip`.
- **FR-30 [V1]** The agent SHALL execute the serve order: geocode (if address) -> nws_alerts ->
  (firms_bbox + wfigs_incidents + wfigs_perimeters + hrrr_weather in parallel) -> pack E ->
  quote -> fetch W -> [V2: spread_run] -> model_infer -> policy -> ActionCard -> LLM brief -> deliver.
- **FR-31 [V1]** The agent SHALL run watch mode on a 5-10 min cron over the book of sites and ask
  mode on demand, using the same tools.
- **FR-32 [V1]** Agent "memory" SHALL be limited to durable logs and per-site state (last poll,
  last card, re-poll cadence); the LLM context SHALL NOT be the source of truth for any number.
  **[NEW DECISION: memory is explicitly scoped to a per-site state store + append-only logs; there is
  no free-form long-term LLM memory in V1.]**
- **FR-33 [V1]** On any tool failure, the agent SHALL retry with backoff and, if unrecoverable, emit
  a degraded card with explicit flags rather than fabricating the missing input.

### 5.8 ActionCard generation

- **FR-34 [V1]** The action SHALL be produced by a deterministic policy table (5.9), not the LLM.
- **FR-35 [V1]** The card SHALL populate all applicable schema fields (4.4), including citations,
  product times, W sources, versions, sigma, and flags.
- **FR-36 [V1]** The LLM brief SHALL copy card fields and citation URLs only; a validator SHALL reject
  any brief introducing a new number or altering CAP severity.

### 5.9 Policy / routing

- **FR-37 [V1]** The policy SHALL be a versioned, inspectable table (tuned with a design partner),
  not hidden in the LLM. Baseline policy (V1):
  - No FIRMS persistence, no WFIGS, no Red Flag -> `no_action` (or `monitor` if SPC outlook high).
  - FIRMS without WFIGS -> `monitor` + `perimeter_unofficial`; never `prepare` on a single urban
    hotspot (developed LCMS).
  - WFIGS dist > 15 km and low `y_hat` -> `monitor`.
  - Dist ~4-10 km + Red Flag + high fuel W + moderate `y_hat` -> `prepare`.
  - Dist small or high `y_hat` + high access/housing -> `protect_asset` / `evacuate_site`, without
    downgrading a NWS warning.
  - Fire contained/gone and site was in play -> `inspect_after`.
- **FR-38 [V2]** The policy SHALL additionally key on ETA: `evacuate_site` when ETA < X h with high
  confidence; `prepare` when ETA in 24-72 h; `monitor` beyond; high `eta_sigma_hours` SHALL prevent
  overconfident `evacuate` and SHALL set `no_ros_high_sigma`.
- **FR-39 [V1]** Delivery routing SHALL support per-site channels (Slack/email) and SHALL escalate on
  `protect_asset`/`evacuate_site`.

### 5.10 Citation and provenance

- **FR-40 [V1]** Every number in the card SHALL carry a citation (source, URL, fetched_at) or a W
  source (field, source_url, vintage, confidence). No uncited numbers.
- **FR-41 [V1]** The system SHALL log every tool request/response JSON, `model_version`,
  `policy_version`, and (V2) `spread_field_version` for audit and later outcome capture.

### 5.11 Alerting integrity

- **FR-42 [V1]** The system SHALL never suppress a CAP alert and SHALL surface it even when `y_hat`
  is low.
- **FR-43 [V1]** The system SHALL de-duplicate repeated cards for the same site/incident/state within
  a configurable window while still escalating on state changes.

### 5.12 Response Support Agent

- **FR-44 [V1]** The system SHALL auto-trigger the Response Support Agent in parallel when the main
  prediction agent emits `protect_asset` or `evacuate_site` for a site.
- **FR-45 [V1]** The Response Agent SHALL fetch the role J W fields (2.2.2) from the same cache (or
  trigger an on-demand fetch if not yet cached), and SHALL perform one fresh live query: USGS gage
  `daily_discharge_cfs` for the nearest gage to the site.
- **FR-46 [V1]** The Response Agent SHALL compile a **ranked water source list** for the site by
  aggregating: (a) nearest stream/flowline with live discharge, (b) nearest waterbody name/distance,
  (c) wetlands within 100 m and their acreage, (d) municipal water service presence, (e) well density,
  (f) nearest wastewater plant distance, (g) high-hazard dams within 10 km. Each source SHALL be rated
  for availability using a deterministic function of `discharge_cfs`, `drought_category`, and
  `surface_water_permanence_pct`.
- **FR-47 [V1]** The Response Agent SHALL produce an **access route assessment** from
  `nearest_major_road_class/name`, `nearest_road_class/surface/distance`, and
  `roads_within_500m_count`, ranked by road class and surface type.
- **FR-48 [V1]** The Response Agent SHALL estimate fire-station ETA as `nearest_fire_station_distance_m`
  divided by a configurable average truck speed (default 60 km/h on paved roads, 30 km/h on unpaved),
  marked as an estimate (never a routing API result in V1).
- **FR-49 [V1]** The Response Agent SHALL produce a **hazmat priority list** from `nearest_hazardous_
  facility_distance_m/name`, `nearest_rcra_tsd_distance_m`, `nearest_ust_facility_distance_m`,
  `ust_facilities_within_1km_count`, `nearest_gas_pipeline_distance_m`, `nearest_petroleum_pipeline_
  distance_m`, and `nearest_transmission_line_distance_m`, ranked by distance, and assigned a priority
  tier (critical: < 500 m; high: < 2 km; medium: < 5 km).
- **FR-50 [V1]** The Response Agent SHALL flag **environmental response constraints**: if
  `intersects_critical_habitat = true`, mark the site as "retardant restricted" with species;
  if `intersects_protected_area = true`, record designation and manager as "coordinate first."
- **FR-51 [V1]** The Response Agent SHALL record the **responsible agency** from
  `surface_management_agency` and include it in the ResponseCard and brief.
- **FR-52 [V1]** The Response Agent SHALL assess **comms availability** from `mobile_5g_coverage_class`
  and `nearest_antenna_structure_distance_m/height_m` and include it in the ResponseCard.
- **FR-53 [V1]** The Response Agent SHALL include `nearest_airport_name/distance_m` as the air tanker /
  helicopter staging field, never computing flight time (distance only).
- **FR-54 [V1]** The Response Agent SHALL emit a **ResponseCard** (schema 4.5) delivered in the same
  Slack/email thread as the ActionCard, with a separate version field `response_card_version`.
- **FR-55 [V1]** Every number in the ResponseCard SHALL carry a citation; the LLM may add a prose
  summary brief copying fields only; no LLM-invented distances, names, or ratings.
- **FR-56 [V2]** The Response Agent SHALL generate an **AOI water map** by querying Mireye at a dense
  grid across the incident AOI and returning a spatially referenced list of water sources within the
  fire risk zone — useful for aerial drop planning. Bounded by the same max-cell / latency cap as the
  main AOI enrichment.
- **FR-57 [V2]** The Response Agent SHALL use a road-network graph (OSM, processed offline) for
  routing rather than the distance proxy, giving a more accurate truck ETA.

---

## 6. Data and Model Requirements

### 6.1 Labels (y)

- **Gold label (V1 target):** site inside the **final** perimeter within 72 h of t0.
  - Source: **MTBS** burned-area boundaries (public domain, ~1 yr lag, large fires only: >1000 ac
    West / >500 ac East) as gold; **final/YTD WFIGS** perimeters for recent seasons (smaller,
    noisier). CAL FIRE FRAP historical SHALL NOT be ingested as live E.
- **Arrival-time label (V2 target):** per-site fire arrival time derived from **perimeter
  time-series** (successive operational perimeters / IR cycles), reduced to hours-to-front and
  `P(burn by T{24,48,72})`. The V1 classifier is the degenerate projection `P(arrival < 72 h)`.
- **Leakage rule:** `y` SHALL NOT be almost-a-feature. Do not use burn-year as a feature to predict
  "will burn"; do not leak the final perimeter into E.

### 6.2 Training data

- Construct samples `(site, t0, event_id)` near historical fires.
- **E_hist:** FIRMS/WFIGS **as of t0** (no final-perimeter leakage); HRRR archive/reanalysis wind at
  t0 for the ROS feature and (V2) the engine.
- **W_hist:** Mireye-like values at vintage t; reconstruct or **drop/mask** NDVI/drought that cannot
  be restored to the event date (never use 2026 NDVI on a 2018 fire); DEM/FHSZ treated as ~stable.
- **Sampling:** all positives in a buffer; hard negatives 2-20 km; easy negatives in other
  ecoregions.
- **Honest data ceiling for learned spread:** public next-day spread SOTA (WSTS+ Res18-UNet) is
  **~0.46-0.47 AP** (logistic baseline ~0.29 AP). This is the explicit reason to **delegate** the
  engine and learn only calibration; the SRS SHALL NOT plan a mask-to-future-mask learned spread
  model.

### 6.3 Features

- **W feature vector for the prediction model** (`mireye_into_model.md`): roles A-I from the
  ~75-85-field **Fire Intelligence Set** (2.2.2), encoded as typed features — scaled numerics +
  binaries + ordered/unordered categorical embeds + missing masks; citations and vintages travel
  beside the vector, not inside it. The set is curated for fire *prediction* signal (fuel, terrain,
  fire-weather climatology, hazard priors, ignition sources, WUI exposure, access/egress, compounding
  hazards, firefighting water). Feature-selection/regularization prunes within this set during
  training; per-role ablations (6.5) verify each role earns its place.
- **W fields for the Response Agent** (role J in 2.2.2): fetched from the same cache but used only
  by the Response Agent (not in the prediction model's feature vector): waterbody/flowline proximity,
  wetlands count/acreage, dams, airport, comms (5G/antenna), critical habitat, protected area
  designation/manager, wastewater plant, sewer service, OSM substation. These are **deterministic
  look-up fields** — ranked and formatted by code, not fed into `h_fire`.
- **E features:** `rflag`, FIRMS counts/FRP by radius, `dist_perim_m`, `acres`, `containment`,
  `hours_since_discovery`, `perimeter_unofficial`, and `wind_ros_ellipse` (V1).
- **HRRR:** 10 m wind U/V, temp, RH (V1 for the ROS feature; V2 for the engine).
- **LANDFIRE (V2):** FBFM40 fuel, canopy (CBD/CBH/CC/CH), DEM - engine inputs, not model features.
- **X (V3, optional):** pre-event EO embedding (AlphaEarth 64-D or Prithvi pooled vector) as a
  cached vintage fuel feature; never a live t0 chip (smoke-blocked, too slow).

### 6.4 Model stages

Behind the stable `model_infer` contract, complexity increases only when the prior rung is beaten:

1. **V1 - MLP/logistic on `[g(W); E]`** (E includes `wind_ros_ellipse`). Target = P(in perimeter
   72 h). Calibrated (isotonic/Platt).
2. **(Research) concat `[pool(f(X)); g(W); E]`** with a frozen EO embedding (V3 gate).
3. **(Research) FiLM:** `MLP(W) -> (gamma, beta)` on vision/E features.
4. **(Research) cross-attention** only if it beats FiLM by a margin worth the cost.
5. **V2 - W-calibration head** on the delegated arrival field:
   `h_fire(raw_field_at_site, Mireye W, E) -> calibrated ETA, P(burn<=T), sigma`.

Invalid approaches (explicitly excluded): serialize W to text -> EO hypernetwork/LoRA; learn
propagation from scratch; mask-image inpainting as spread.

### 6.5 Evaluation protocol

- **Splits:** entire **event IDs** held out; plus **HUC-6** and **state** clustered holdouts.
  Random pixels from one fire in train and test = leakage and is forbidden.
- **Baselines:** V1 = distance-to-current-perimeter + crude ROS, and E-only / W-only / X-only;
  V2 = **raw ELMFIRE** (hard operational bar) + exposure/climatology.
- **Metrics:** PR-AUC / Average Precision, **Brier + calibration** (reliability), and (V2)
  **arrival-time skill** (e.g. MAE of ETA, skill vs raw field) and probability calibration of
  `p_burn_by_T`.
- **Mandatory falsification probes at every rung:** (1) **random-W collapse** - shuffle W; if the
  metric holds, W is decorative; (2) **E-only / W-only / X-only** ablations - if E-only equals the
  full model, admit it is a distance-to-fire wrapper; (3) **per-role W ablation** - drop each role
  (A-I) in turn to confirm the rich set earns its keep and to prune roles that add nothing (this is
  the honest counterweight to fetching ~60 fields: generous fetch, disciplined model); (4)
  temporal/leakage audit (burn-history/NDVI vintage cannot leak the label).
- **Scoreboards separate:** research = skill/calibration vs baseline on holdout; product = sites,
  days unattended, zero invented numbers, human preference. Never report mIoU as product success or
  Slack adoption as research skill.

### 6.6 Research contribution statement and falsifier

- **Contribution (single sentence):** *Mireye-fuel-conditioned calibration of an operational
  wildfire-spread model (ELMFIRE) improves site-level fire arrival-time and burn-probability skill
  versus the raw operational model, on event-, HUC-, and state-held-out fires.*
- **Falsifier:** if the W-conditioned calibration does **not** beat raw ELMFIRE on held-out fires
  (and random-W does not hurt), then W adds nothing on top of physics; ship the delegated model
  alone and retract the "baked-W" claim. (V1 sub-claim H1 has the same shape against the
  distance+ROS / E-only baseline; if H1 fails, ship the agent as honest orchestration and pause the
  baked-W story.)

---

## 7. Non-Functional Requirements

### 7.1 Performance / latency

- **NFR-1 [V1]** Ask-mode interactive query SHALL return an ActionCard within a target of **<= 30 s**
  p95 (dominated by external fetches + one model call). **[NEW DECISION: 30 s p95 interactive target
  set; not specified in source docs - tune with the design partner.]**
- **NFR-2 [V1]** Watch-mode SHALL poll every 5-10 min per site and complete a full book-of-sites
  cycle within the poll interval, batching Mireye and parallelizing E.
- **NFR-3 [V2]** Spread-engine runs are **asynchronous per incident** (minutes-hours per ensemble);
  cards SHALL use the latest completed field and SHALL flag when the field is stale relative to E.

### 7.2 Scalability

- **NFR-4 [V2]** Cost SHALL scale with the number of **active incidents** (tens nationally at peak),
  not with the number of sites or town area: one ELMFIRE run per incident serves all nearby sites;
  sampling the field per site is free.
- **NFR-5 [V1/V2]** AOI point-tiling SHALL be bounded by a max-cell limit chosen for **latency and
  relevance** (per-poll cycle time, coherent AOI), NOT for credit survival - credits are abundant.
  V2 MAY use a **denser AOI enrichment grid** than a cost-minimizing design would allow, when it
  improves field quality; coarse-grid W caching is used so only vintage-dynamic roles refresh.

### 7.3 Cost / credits budget

- **NFR-6 [V1]** Mireye credits are abundant, so the budget is a **quality envelope, not a survival
  cap**: V1 SHALL operate within **~300,000-400,000 credits**. Cost is dominated by one-time
  enrichment: ~55-65 credits/site for the Fire Intelligence Set (+300/site if optional parcel
  geometry), mostly static and cached long; only vintage-dynamic roles (NDVI/drought) refresh, at a
  few credits/site/refresh. E and HRRR are free. A book of ~1,000-5,000 sites fits comfortably; a
  live credit tracker SHALL surface spend against the envelope, and `quote` SHALL precede every
  `fetch` (reproducibility + waste-avoidance, not rationing).
- **NFR-7 [V2]** V2 SHALL spend **as many credits as quality needs** (denser AOI enrichment, richer
  refresh). Spread compute (ELMFIRE ensembles) is the primary V2 COGS: budget CPU per incident-run *
  ensemble-size * active incidents; ensemble size is tuned for **arrival-time sigma quality**, not to
  minimize spend, within an overall compute budget that is provisioned rather than rationed.
- **NFR-8 [V1]** OpenAI tokens are COGS: briefs are short, copy-only; token spend SHALL be bounded
  per card.
- **NFR-9 [V1]** Pricing model is **per site-month of watch**; owner/parcel fields, if ever offered,
  are a separate slow SKU.

### 7.4 Reliability

- **NFR-10 [V1]** External calls SHALL use retries with backoff, token buckets (FIRMS quota is
  UNVERIFIED), and User-Agent + contact (NWS). A failed feed SHALL degrade the card with flags, never
  fabricate.
- **NFR-11 [V1]** The watch loop SHALL be idempotent per (site, poll) and resumable after crash from
  durable state.

### 7.5 Security / privacy

- **NFR-12 [V1]** API keys (Mireye, FIRMS MAP_KEY, OpenAI) SHALL be stored as secrets, never logged.
- **NFR-13 [V1]** The book of sites is customer data; access SHALL be authenticated and scoped;
  parcel-owner data SHALL NOT be fetched or stored.

### 7.6 Provenance / auditability

- **NFR-14 [V1]** Every card SHALL be fully reconstructable from logs: tool JSONs, versions
  (`model_version`, `policy_version`, `spread_field_version`), and product times.
- **NFR-15 [V1]** `dataset_vintage`/`fetched_at` SHALL be recorded and surfaced; `ndvi_current` SHALL
  never be described as live greenness.

### 7.7 Licensing compliance

- **NFR-16 [V2]** The spread engine SHALL run **out-of-process behind `spread_run`**; ELMFIRE
  (EPL-2.0) modifications, if any, SHALL be disclosed per EPL; Cell2Fire (GPL-3.0), if ever used,
  SHALL be isolated behind a process/network boundary with legal review. No copyleft engine source
  SHALL be linked into proprietary model/agent code.

### 7.8 Hard boundary (safety)

- **NFR-17 [V1]** The system SHALL never publish an official perimeter as truth; it SHALL always emit
  an **arrival estimate with sigma**. It SHALL never downgrade or suppress a NWS/CAP warning. High
  sigma SHALL prevent overconfident `evacuate`.

---

## 8. V1 vs V2 (end-to-end)

### 8.1 V1 (MVP) - end-to-end, exactly what ships

**In one paragraph.** V1 is an unattended, cited, **quality-first** fire-week copilot for a book of
point/parcel US sites, operating within a ~300-400K Mireye credit envelope (credits abundant). On a
5-10 minute watch loop (and on-demand ask), the agent fetches live E - NWS CAP Red Flag, SPC outlook,
FIRMS VIIRS/MODIS hotspots + FRP in 5/10/20 km radii, WFIGS incidents and perimeters (geodesic
distance in code), and HRRR wind/RH/temp - and computes a wind-projected ROS-ellipse feature; it
quotes and fetches the **~60-field Mireye Fire Intelligence Set** (fuel, terrain, fire-weather
climatology, hazard priors, ignition sources, WUI exposure, access/egress, compounding hazards,
firefighting water; roles A-I in 2.2.2), plus optional parcel-geometry footprint precision, typed +
masked and vintage-tracked; it calls `model_infer`, a calibrated MLP on `[g(W); E]` whose target is
`P(site in final perimeter within 72 h)` with sigma and a distance+ROS `baseline_y`; a deterministic,
versioned policy table converts the score + E into a closed action (monitor / prepare / protect_asset
/ evacuate_site / inspect_after / no_action) - and here the rich W directly raises quality, because
access/egress, fire-station distance, WUI density, and compounding-hazard fields feed
protect/evacuate reasoning rather than a bare distance-to-fire; the agent emits a fully cited
ActionCard and an LLM brief that copies fields only, delivered to Slack/email, with every tool JSON
logged; and when action is `protect_asset` or `evacuate_site`, a **Response Support Agent fires in
parallel** and delivers a **ResponseCard** in the same thread — ranked water sources (streams by live
USGS discharge, lakes, wetlands, municipal, dams), access routes ranked by road class/surface,
fire-station ETA estimate, hazmat priority list, environmental retardant constraints, responsible
agency, and comms availability — all cited, no LLM-invented numbers. Honest metrics are reported separately for research (PR-AUC/Brier vs distance+ROS and
E-only/W-only plus per-role W ablations on event+HUC/state holdout, random-W collapse as a gate) and
product (N sites, days unattended, zero invented numbers, human prefers the card).

**Explicitly OUT of V1 (deferred to V2):** the delegated spread engine (ELMFIRE) and arrival-time
target; LANDFIRE; the Mireye-W calibration head; incident-first AOI and dense AOI enrichment; the EO
embedding on-ramp; RAWS augmentation; outcome capture. **Excluded entirely (not a V2 promise):**
live EO imagery on the hot path (smoke-blocked), learned fire-spread from scratch, `parcel_owner`,
`/v1/ask` for facts, and a modeled debris-flow head.

### 8.2 V2 - end-to-end

**In one paragraph.** V2 makes fire spread the spine without ever learning the physics. Geometry
becomes **incident-first**: for each active WFIGS incident the system builds a wind-projected buffer
around the perimeter clipped to burnable fuel, runs a **delegated operational engine (ELMFIRE via
Cloudfire, EPL-2.0, out-of-process behind `spread_run`)** fed by **LANDFIRE** fuel/canopy/DEM + HRRR
weather + WFIGS/FIRMS ignition, and produces a per-cell arrival-time / P(burn) field (with an
ensemble giving per-site sigma); it then intersects the book of sites with the field. `h_fire`
becomes a **Mireye-W-conditioned calibration head** on the raw field, outputting per-site
`eta_hours`, `eta_sigma_hours`, and `p_burn_by_T{24,48,72}` alongside the still-valid `P(burn by
72 h)` projection - always framed as "arrival estimate with sigma," never an official perimeter, never
downgrading NWS. The policy keys on ETA (evacuate when ETA < X h and confident; prepare at 24-72 h;
monitor beyond; high sigma blocks overconfident evacuate). The research eval upgrades to the stronger
claim: W-conditioned calibration vs **raw ELMFIRE** on event/HUC/state held-out fires, with the same
ablation gates. Same agent, same `model_infer`/ActionCard contract; cost scales with active
incidents, not sites, because LANDFIRE and HRRR are free and one run serves all nearby sites.

**Response Agent upgrades in V2.** The Response Agent gains two capabilities: (a) an **AOI water
map** — dense Mireye grid over the incident AOI returns a spatially referenced list of all water
sources inside the fire risk zone, useful for aerial drop planning and crew positioning; (b)
**OSM road-network routing** replaces the distance-proxy ETA with a proper truck-route estimate
using an offline-processed road graph. Both use no trained model; quality improves through richer
spatial coverage and more accurate routing math.

**Optionals folded into V2 (because budget is abundant and quality is the goal).** Per the mandate
that optionals integrate into V2, the following move from "someday V3" into the V2 quality scope,
each still gated by an honest condition: (a) **EO embedding on-ramp** - AlphaEarth 64-D (zero-GPU)
as a precomputed vintage fuel-texture feature concatenated into `g(W)`, added *only if* a per-role
ablation shows a fuel-continuity/texture hole the scalar W misses; (b) **dense AOI enrichment** -
grid-sample the Mireye Fire Intelligence Set across the incident AOI (not just at book sites) to
sharpen the calibration field, now affordable; (c) **outcome capture** - log did-they-prepare/
evacuate/claim into a private Y stream for future supervision, *if* the design partner shares
outcomes; (d) **RAWS/MesoWest point wind/RH augmentation** - included *only if* the license verifies
compatible (still UNVERIFIED; pursued during V2, not assumed). Items that genuinely depend on a
working V2 engine - the **distilled fast surrogate** (needs our own ELMFIRE runs to train against)
and a **fine-tuned EO encoder** - remain V3 and are NOT pulled forward, because doing so would be
dishonest sequencing, not a budget question.

### 8.3 Comparison table

| Capability | V1 (MVP) | V2 | V3 (only if prior rung works) |
|---|---|---|---|
| Geometry | point / parcel (book of sites) | + incident-first AOI (buffer around active perimeter), intersect sites | multi-incident national triage view |
| Mireye W | **~60-field Fire Intelligence Set** (roles A-I) + optional parcel geometry | same set + **dense AOI grid enrichment** (sample W across the incident AOI) | - |
| Live E feeds | CAP, SPC, FIRMS, WFIGS, **HRRR** | same + **RAWS** point wind/RH *if license verifies* | - |
| Spread signal | wind-projected ROS ellipse **feature** | **delegated ELMFIRE arrival field** (per incident) | distilled fast surrogate of ELMFIRE |
| Fuel data (engine) | Mireye W only (calibration/context) | + **LANDFIRE** (engine fuel/canopy/DEM) | - |
| EO imagery | none (X=null) | **precomputed AlphaEarth 64-D embedding** as vintage fuel feature *if per-role ablation shows a hole* | fine-tuned EO encoder (Prithvi), never live chip |
| Model target | `P(in perimeter 72 h)` + sigma | **arrival time / P(burn by T{24,48,72})** + sigma | damage-class head (if labels exist) |
| Model form | calibrated MLP on `[g(W);E]` | **W-conditioned calibration head** on raw field (+opt. concat AlphaEarth) | FiLM/cross-attn EO fusion |
| Agent / ActionCard | full, deterministic policy, cited | same contract, ETA-keyed policy, + **outcome capture -> private Y** *if partner shares* | - |
| Response Support | **ResponseCard** on protect/evacuate: ranked water (USGS live), access routes, hazmat priority, constraints, agency, comms | + **AOI water map** (dense grid) + **OSM road-network ETA** | - |
| Research claim | H1: W beats distance+ROS/E-only | H2: W-calibration beats **raw ELMFIRE** | distillation / EO-fusion studies |
| Cost driver | Mireye credits/site (~75-85 cr) + OpenAI (both abundant) | + spread compute per **incident** (provisioned) | + GPU for surrogate/encoder |

---

## 9. Optional / Stretch

Because credits and compute are abundant, "optional" here means **gated by an honest engineering
condition, not by budget**. Several former V3 items are pulled into **V2** (see 8.2/8.3); the rest
remain V3 only because they genuinely depend on a working V2 or on data we do not yet have. Each item
ships only if its condition holds.

**Folded into V2 (quality scope):**

- **9.1 EO embedding on-ramp** - precomputed **AlphaEarth 64-D** (zero-GPU) concatenated into `g(W)`
  as a vintage fuel-texture feature. *Only if* a per-role W ablation reveals a fuel-continuity/texture
  hole the ~60 scalar fields miss. Always a precomputed embedding, never a live t0 chip (smoke).
- **9.2 Dense AOI enrichment** - grid-sample the Fire Intelligence Set across the incident AOI to
  sharpen the calibration field. Affordable at this budget; bounded by latency, not credits.
- **9.3 RAWS / MesoWest augmentation** of point wind/RH. *Only if* the license verifies compatible
  (UNVERIFIED today); pursued during V2, never assumed.
- **9.4 Outcome capture** (did they prepare/evacuate/claim?) -> private Y stream for future
  supervision. *Only if* the design partner shares outcomes.

**Remain V3 (depend on a working V2 or on missing data - NOT a budget question):**

- **9.5 Distilled fast surrogate** of ELMFIRE (ConvLSTM / neural CA / FNO) for latency. Requires our
  own ELMFIRE runs as training targets, so it cannot precede V2; never a serve dependency before V2.
- **9.6 Fine-tuned EO encoder** (Prithvi-EO-2.0-300M-TL frozen HLS/SWIR, pooled -> concat -> FiLM).
  *Only if* the frozen AlphaEarth embedding (9.1) wins its ablation and fine-tuning beats it on
  held-out regions. Precomputed features only.
- **9.7 Multi-incident national view** across all active incidents. *Only if* incident-first V2 is
  stable and a customer needs cross-incident triage; still per-site cards underneath.
- **9.8 Additional perils** (e.g. post-fire debris-flow head). *Only if* labels exist; until then,
  alert-only if CAP mentions debris flow; no `h_debris` head is planned.

---

## 10. Risks, Open Questions, and UNVERIFIED Items

### 10.1 Licensing / copyleft

- ELMFIRE is EPL-2.0 (weak copyleft) - compliant only if run out-of-process and modifications
  disclosed. Cell2Fire is GPL-3.0 (strong copyleft) - risky to link into closed SaaS; prefer ELMFIRE,
  isolate Cell2Fire if used, legal review required.

### 10.2 UNVERIFIED data/licenses

- **RAWS / MesoWest** license - UNVERIFIED; not used in V1/V2.
- **FIRMS MAP_KEY numeric quota** - UNVERIFIED; mitigate with token bucket + backoff.
- **"DeepMind FireCast v3 / firecast.deepmind.dev, 89% accuracy"** - UNVERIFIED / likely fabricated;
  **do not integrate or design against it.**
- **FARSITE/FlamMap/Technosylva** 2026 automation/licensing terms - UNVERIFIED; not the pick.
- Gridded operational fuel-moisture specifics beyond what Cloudfire fetches - UNVERIFIED; treat as
  "comes with the weather feed."

### 10.3 Data ceilings and biases

- Learned next-day spread SOTA is <0.5 AP (WSTS+) - reason to delegate, not learn dynamics.
- MTBS is large-fire biased and ~1 yr lagged - the model may underweight small fires; product must
  still `monitor` FIRMS starts.
- FIRMS false positives (flares/industry) - require WFIGS confirm or persistence + non-developed LCMS.
- SNPP VIIRS delivery ends 2026-11-01 - NOAA-20/21 only thereafter.
- FHSZ is CA-centric - mask outside CA, never fake.
- NDVI/W are vintage - never call live; pass vintage into the model and card.
- No RAWS -> ROS uncertainty is high -> high sigma; do not overconfident-evacuate.

### 10.4 Operational burden (V2)

- Running an operational fire model per incident (job queue, ensemble tuning, failure handling, LANDFIRE/
  HRRR pipelines) is real ops burden; this is the main new cost of V2. Async, incident-scoped design
  contains it, but it is a genuine step-up from "orchestration + light model."

### 10.5 Calibration credibility (V2)

- The W-calibration claim needs enough held-out fires to be credible; if too few fires have clean
  perimeter time-series + reconstructable t0 weather, the research claim may be underpowered.

### 10.6 Scope-boundary tension (flagged in source)

- Spread-as-spine flirts with the scoped-out "predict tomorrow's perimeter as truth." Resolution
  (from `plan_fire_architecture_qa.md`): emit **arrival estimate with sigma**, delegate the physics,
  never publish an official perimeter, never downgrade NWS. With that framing the guardrail holds and
  the target change is an extension, not a contradiction.

### 10.7 NEW DECISIONS made in this SRS (for your review)

These are choices this SRS settled that were not explicitly nailed down in the source docs. Each is
consistent with the source docs' direction; flagged so you can override.

1. **Integration seam formalized:** the spread engine is always invoked **out-of-process behind a
   `spread_run` internal API** so copyleft (EPL/GPL) never links into proprietary code (2.6.1, 7.7).
   This is the compliant boundary; the docs discuss the licensing risk but did not name the seam.
2. **OpenAI named as the V1 LLM provider**, with a recommendation to keep it provider-agnostic (4.2).
   The docs say "LLM/OpenAI" loosely; this fixes it for V1 and flags swap-ability.
3. **Interactive latency target = 30 s p95** for ask mode (NFR-1). No number existed in the docs;
   tune with the design partner.
4. **Agent memory scoped** to a per-site state store + append-only logs, with no free-form long-term
   LLM memory in V1 (FR-32). The docs imply this but did not state it as a requirement.
5. **ActionCard schema fully enumerated** with concrete field names/types incl. `eta_sigma_hours`,
   `p_burn_by_T` as a {24,48,72} object, `reasons[]`, `policy_version`, `spread_field_version`,
   `e_product_times[]`, `w_sources[]` (4.4). The docs specified a partial contract; this SRS
   formalized the full schema.
6. **Explicit V1 evacuate/ETA thresholds deferred to policy tuning** rather than hard-coded here
   (FR-37/FR-38). The docs give a policy sketch; concrete cutoffs ("ETA < X h") are left as tunable
   `policy_version` parameters - flagged so you can set X.
7. **AlphaEarth chosen as the first EO on-ramp** (zero-GPU) ahead of Prithvi (V3, 9.5), matching the
   architecture QA recommendation; recorded here as the default ordering.
8. **Ensemble-driven sigma for arrival time** (FR-22) as the V2 uncertainty source; the docs call for
   sigma but did not specify the mechanism - ensemble perturbation is chosen as the default.
9. **Credit posture flipped to abundance (>1,000,000 credits); V1 quality budget = ~300-400K, V2 =
   as needed** (header, 2.6.2, 7.3). The prior SRS/source framing conserved against a 25K/month cap;
   this revision treats credits as a quality envelope, and every "to save credits" rationale is
   rewritten to "for freshness/latency/reproducibility." Override if the real budget differs.
10. **W expanded from an ~11-field allow-list to a curated ~60-field Fire Intelligence Set (roles
    A-I)** (2.2.2, 6.3), with per-role W ablation added as a mandatory gate (6.5) so the generous
    fetch stays honest. Curated for fire signal, not raw cardinality; siting-only fields excluded.
    This is the single biggest quality lever and the main change in this revision.
11. **Optional parcel-geometry footprint precision admitted into V1** (FR-3), one 300-cr parcel field
    per site; `parcel_owner` still excluded for privacy.
12. **Former V3 optionals folded into V2** (8.2, 8.3, 9, P2): AlphaEarth EO embedding, dense AOI
    enrichment, outcome capture, RAWS (conditional). Distilled surrogate and fine-tuned encoder stay
    V3 because they depend on a working V2 engine - an honest-sequencing decision, not a budget one.
13. **Response Support Agent added as a parallel specialized agent** (2.2.5, 4.5, 5.12) triggered on
    `protect_asset`/`evacuate_site`. Uses no trained model; all deterministic code on Mireye W + live
    USGS gage. Role J fields added to the Fire Intelligence Set for this purpose (~20 new fields,
    ~75-85 credits/site total). This agent genuinely addresses the RESPONSE SUPPORT dimension which
    was weak before. V2 adds AOI water map + OSM road-network routing. New decisions within 13: fire
    station ETA uses distance/speed proxy in V1 (OSM routing deferred to V2); water availability is
    a deterministic code function of discharge/drought/permanence; retardant constraint flag is
    boolean from `intersects_critical_habitat`; hazmat priority tiers at 500m/2km/5km (override
    if the design partner wants different cutoffs).

None of these resurrect learned-spread-from-scratch, and none contradict the source docs; they make
under-specified points concrete and apply the abundant-credit, quality-first mandate.

---

## 11. Milestones / Phased Roadmap

Mapped to V1/V2/V3; smallest honest shippable increment first.

| Phase | Timeframe | Ships | Maps to | Done when |
|---|---|---|---|---|
| **P0** | 2-4 wk | Agent works: YAML sites, 5-10 min poll; CAP + FIRMS + WFIGS + **HRRR** + **~75-85-field Fire Intelligence Set** (roles A-J); **mlp-v0** trained on a tiny historical extract (real theta, `model_version`); ActionCard schema; **ResponseCard schema + Response Agent** (water list, access routes, hazmat, constraints, agency, comms) on protect/evacuate; Slack delivery; every tool JSON logged; `/v1/ask` and parcel-owner banned | V1 core | Stranger adds a lat/lng, leaves overnight, gets a cited ActionCard + ResponseCard; zero invented numbers |
| **P1** | Q1 | Policy tuned with design partner; long/short-TTL W cache; optional parcel-geometry footprint; `wind_ros_ellipse` hardened; USGS live gage discharge confirmed reliable; research R1 (MLP on [W;E] vs distance+ROS/E-only, random-W + per-role W ablation gates) | V1 complete | One real fire week used for decisions; H1 pass/fail reported honestly; ResponseCard used by site manager |
| **P2** | next | **LANDFIRE** integrated; **ELMFIRE** service stood up (out-of-process, incident-first AOI); target switched to **arrival-time / P(burn by T)**; **W-calibration head**; folded-in quality optionals: dense AOI enrichment, AlphaEarth embedding (if ablation shows a hole), outcome capture (if partner shares), RAWS (if license verifies); research H2 vs raw ELMFIRE on held-out fires | V2 | Paid site-month watch on arrival-time cards; H2 pass/fail reported |
| **P3** | later | Distilled fast surrogate (trained on our ELMFIRE runs); fine-tuned EO encoder; multi-incident national view | V3 | Surrogate matches ELMFIRE within tolerance at lower latency |

Guardrail: do **not** wait on ELMFIRE, LANDFIRE, RAWS, or an EO encoder to call the agent "working."
If R1/H1 fails, keep the P0/P1 loop as honest orchestration and pause the baked-W marketing.

---

## 12. Acceptance Criteria

### 12.1 V1 is done when (product)

- **AC-1** A stranger can add sites via YAML/CSV, run the watch loop unattended for **>= 7 days over
  >= 10 sites**, and receive cited ActionCards with **zero invented numbers** (audited against logs).
- **AC-2** Every card populates the mandatory schema fields (action, y_hat, sigma, baseline_y,
  citations, e_product_times, w_sources, model_version, policy_version, flags) and passes the
  no-uncited-number validator.
- **AC-3** The LLM brief validator rejects any brief that introduces a new number or alters CAP
  severity (tested with adversarial inputs).
- **AC-4** CAP warnings are never suppressed or downgraded (verified test cases).
- **AC-5** Mireye usage stays within the ~300-400K V1 credit **envelope** (not a survival cap) with
  `quote`-before-`fetch` and role-dependent W caching, tracked by a live credit meter; FIRMS/NWS
  calls respect rate limits and User-Agent.
- **AC-6** In a real fire week, a human operator states they prefer the card to opening browser tabs
  (design-partner sign-off).
- **AC-6b** When a site reaches `protect_asset` or `evacuate_site`, a ResponseCard is delivered in
  the same thread within the p95 latency target, with: at least one ranked water source with a live
  USGS discharge (or an explicit "gage unavailable" flag), a hazmat list (empty if none within 5 km),
  responsible agency, and zero LLM-invented distances or names. A design-partner site manager
  confirms the card is useful for decision-making during a real incident.

### 12.2 V1 research claim (H1) validated / falsified when

- **AC-7** On event + HUC-6 + state held-out splits, the calibrated model on `[g(W); E]` beats the
  distance+ROS baseline and E-only on PR-AUC/AP and Brier, **and** shuffling W measurably hurts.
  -> H1 validated.
- **AC-8** If skill ~ 0 vs baseline or random-W does not hurt -> H1 **falsified**: ship the agent as
  orchestration, pause the baked-W story (recorded honestly).

### 12.3 V2 is done when

- **AC-9** ELMFIRE runs out-of-process per active incident, fed by LANDFIRE + HRRR + WFIGS/FIRMS,
  producing a versioned arrival-time/P(burn) field with per-site ensemble sigma, reused across nearby
  sites.
- **AC-10** Cards emit `eta_hours`, `eta_sigma_hours`, `p_burn_by_T{24,48,72}` framed as "arrival
  estimate with sigma," with ETA-keyed policy and high-sigma evacuate suppression.
- **AC-11** Cost scales with active incidents, not site count (measured), and licensing seam
  (out-of-process engine) is verified.

### 12.4 V2 research claim (H2) validated / falsified when

- **AC-12** On event/HUC/state held-out fires, W-conditioned calibration beats **raw ELMFIRE** on
  arrival-time skill and `p_burn_by_T` calibration, with random-W collapse as a passing gate.
  -> H2 validated (the genuine contribution).
- **AC-13** If W-calibration does not beat raw ELMFIRE (and random-W does not hurt) -> H2
  **falsified**: ship the delegated model alone; retract the baked-W claim.

---

End of SRS.
