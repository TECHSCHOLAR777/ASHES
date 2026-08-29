# Build decisions not settled by the SRS

One line per decision, in build order.

- [2026-08-27] Python 3.12.13 used instead of 3.11 because it is the newest 3.x available on this
  machine via the `py` launcher (3.14 is installed as default but several pinned scientific
  packages lack wheels for it); 3.12 satisfies the SRS's "3.11+" floor.
- [2026-08-27] FIRMS_MAP_KEY was not supplied with the other keys. The FIRMS client degrades to a
  `firms_unavailable` flag rather than blocking the build; see `config/README_MISSING.md`.
- [2026-08-27] SLACK_BOT_TOKEN and SMTP_* were not supplied. Both delivery modules run in
  log-only mode by default (write the payload to the JSONL log, return a synthetic thread id);
  see `config/README_MISSING.md`.
- [2026-08-27] Model training is deferred to the user's own machine/CI with a GPU-optional CPU
  path; the synthetic-data smoke test in Step 18 trains a small MLP on CPU only (500 samples,
  hidden layers capped at 128 units), which finishes in seconds and never touches a GPU.
- [2026-08-27] Ordered categorical fields (`drought_category`, `fema_flood_zone`,
  `soil_drainage_class`) are encoded with a fixed lookup table baked into `w_encoder.py` rather
  than fit from data, since V1 has no historical training corpus yet; the mapping is documented
  inline where defined.
- [2026-08-27] Verified the live Mireye API against the real keys and found three shapes
  the SRS did not specify, all now implemented in `src/clients/mireye.py`: (1) an explicit
  field list over 50 fields is rejected with `fields_too_many` (presets are exempt) - the
  client now chunks any field list at 50 and sums quoted credits across chunks; (2) `/v1/fetch`
  and `/v1/fetch/batch` return a nested `{"fields": {name: {value, confidence, source_url,
  dataset_vintage, ...}}}` shape, not a flat dict - the client flattens this into the
  `field`/`field_confidence`/`field_source_url`/`field_vintage` convention the rest of the
  codebase (`w_encoder.py`, `response_agent.py`) already used; (3) batch quoting takes
  `{"locations": <count>}` (an integer), not the actual coordinate list, since quote cost is
  location-count-dependent, not location-specific; and `/v1/fetch/batch` itself takes
  `{"locations": [{"lat","lng"}, ...]}`, not `{"coords": [...]}`.
- [2026-08-27] The live `/v1/geocode` response has no `confidence` or `range_interpolation`
  field; it returns `accuracy` (0-1 float) and `accuracy_type` (e.g. `rooftop`,
  `range_interpolation`) from the underlying geocodio provider. `GeoPoint.confidence` is set
  to `accuracy_type` directly, and `range_interpolation` is set to `True` whenever
  `accuracy_type` is present and not in `{rooftop, point}`.
- [2026-08-27] Mireye rate limit set to 300 rpm/key (not the SRS's generic 60 rpm): the
  user confirmed the three issued keys are on Mireye's Growth plan ($99/mo, 120,000 credits,
  300 rpm). Three keys round-robin to an effective ~900 rpm budget.
- [2026-08-27] HRRR's native grid is Lambert Conformal - `latitude`/`longitude` come back
  as 2D curvilinear coordinates, not a 1D axis, so `xarray`'s `.sel(..., method="nearest")`
  cannot build a pandas index on them (confirmed against a live fetch: "Could not
  automatically create PandasIndex for coord 'latitude' with 2 dimensions"). Nearest
  neighbor is instead found by brute-force squared-distance argmin over the full ~1.9M-cell
  CONUS grid (`_nearest_grid_index` in `hrrr.py`), which is fast (<10ms) with numpy.
- [2026-08-27] Raising `--workers` from 12 to 24 to try to spend more Mireye credit in the
  same wall-clock window backfired: it overwhelmed Mireye's own backend, not the client's
  rate limiter. Live logs showed 695 read-timeouts, 69 HTTP 500s, and 46 HTTP 429s in ~7
  minutes at 24 workers, versus a clean run at 12 workers - each of the 61 fields in a
  fetch does its own geospatial lookup server-side, and that clearly doesn't scale linearly
  with client-side concurrency past some point. Net effect at 24 workers was *worse*
  throughput (17 samples/15min, mostly wasted retries) than at 12 (34 samples in 512s
  cleanly). Reverted to 12 workers, the last concurrency level validated to complete
  requests cleanly rather than mostly retry-and-fail.
- [2026-08-27] `build_training_set.py`'s first concurrent version shuffled individual
  sample tasks (not just fire order) before submitting them to the thread pool, on the
  theory that interleaving fires kept early progress diverse. In practice this defeated the
  one optimization that matters most at scale: samples from the same fire share a single t0
  (noon UTC on ignition date, `T0_HOUR_UTC`), so grouped together they cost one real HRRR
  fetch per fire; shuffled, every sample lands on a different worker's turn at the (globally
  serialized, cfgrib-thread-safety-required) HRRR parse lock, so it collapsed to roughly one
  HRRR cold-fetch per SAMPLE. Confirmed live on the 2015-2023 CONUS run: 8 samples in 6.5
  minutes with 12 workers (~49s/sample, barely better than the unparallelized 34s/sample
  baseline). Fixed by keeping tasks grouped by fire (only the fire order is shuffled) so the
  thread pool naturally assigns several workers to the same fire's samples at once - one
  pays the real HRRR cost, the rest hit the in-memory cache almost immediately.
- [2026-08-27] MTBS's WFS returns geometry in its native projected CRS (meters), not
  lat/lng degrees, unless `srsName=EPSG:4326` is passed explicitly - the scalar
  `burnbndlat`/`burnbndlon` properties are correctly in degrees regardless, which masked
  this in earlier spot-checks that only printed those properties. A live 1000-fire training
  run caught it immediately: every sample's positive/hard-negative point (derived from the
  unprojected geometry) was a raw meter coordinate fed to Mireye as "lat/lng", and every
  single one came back `coord_out_of_bounds`. Fixed by adding `srsName: "EPSG:4326"` to
  `MTBSClient.get_fires_in_bbox`'s WFS request params; verified the returned coordinates
  are real CA/OR degree values after the fix.
- [2026-08-27] The SRS marks FIRMS MAP_KEY quota "UNVERIFIED" (10.2), but it is directly
  checkable: `GET https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=...`
  returns `{"transaction_limit": 5000, "current_transactions": N, "transaction_interval":
  "10 minutes"}` - confirmed live 2026-08-27 (58 transactions already used on the shared
  key at check time). `FIRMSClient` now self-throttles against this real 5,000/10min budget
  with a sliding window (mirroring `MireyeClient`'s per-key limiter) and exposes
  `get_quota_status()`. This does not change SRS 10.2's status marker (that's the
  document's own text), but the client no longer treats the number as unknowable.
- [2026-08-27] FR-29/FR-30's "typed tools" the agent orchestrates are implemented as a fixed
  deterministic Python call sequence (`build_action_card` in `main_agent.py`), not as OpenAI
  function-calling tools the LLM chooses to invoke. The SRS's own hard constraints (2.6.3,
  3.1) require the exact fixed order regardless of what an LLM might otherwise decide, and
  forbid the LLM from computing any number - a deterministic wrapper the LLM cannot deviate
  from satisfies that more robustly than exposing the same tools via function-calling and
  then constraining the LLM's choices after the fact. The LLM's only role remains what FR-36
  specifies: writing a copy-only brief from the finished ActionCard.
- [2026-08-27] Added `scripts/credit_report.py` (not explicitly named in the SRS, but
  required by NFR-6/AC-5's "a live credit tracker SHALL surface spend against the envelope"):
  sums `credit_usage` log records and reports spend against the 300K-400K V1 envelope.
- [2026-08-27] `scheduler.add_job(..., next_run_time=None)` in the original `watch_loop.py`
  does not mean "schedule normally" - APScheduler treats an explicit `None` as "start this
  job paused, with no automatic runs ever." A live run confirmed this: exactly one poll
  cycle happened (the manual `tick()` call immediately after `scheduler.start()`), and the
  scheduled interval job never fired again even after 10+ minutes. Fixed by omitting the
  argument so APScheduler computes its own next-run time (now + interval) as intended.
- [2026-08-27] `cfgrib`/ecCodes is not thread-safe: a live 5-site watch-loop poll cycle
  (each site fetching HRRR in its own worker thread, per FR-30's parallel E-fetch) produced
  "fatal flex scanner internal error--end of buffer missed" and ecCodes parser errors from
  concurrent GRIB2 parsing. First fix (a single global parse lock) made the corruption stop
  but exposed the underlying performance problem: every site independently re-downloaded
  and re-parsed the same HRRR grid for the same hour, serialized behind the lock, so a
  5-site poll cycle took **over an hour** end to end - far outside NFR-2's 5-10 min target.
  Real fix: `HRRRClient` now caches the fully `.load()`-ed (materialized, not lazy) parsed
  dataset per run-hour and shares it across sites requesting the same hour; only the first
  site pays the real fetch/parse cost, everyone else gets an in-memory cache hit. `.load()`
  inside the parse lock also closes a subtler thread-safety gap: lazy xarray/cfgrib
  DataArrays re-enter the non-thread-safe reader on first access, so later concurrent
  `.isel()` calls from other threads (safe on plain numpy) would otherwise still race on
  cfgrib's C-level state. Verified this actually fixes the cycle time on a rerun.
- [2026-08-27] SPC's Day-1 fire weather outlook has no stable plain-text product URL: the
  originally assumed `fwdy1.txt` 404s, and SPC's "FWD" (Fire Weather Outlook Discussion)
  text product is not exposed per-office (KWNS) on `api.weather.gov/products`. Pointed the
  client at `fwdy1.html`, the real serving page, which is a graphical map viewer - keyword
  extraction on it is best-effort and biased toward false negatives, never a fabricated
  "elevated" reading. This is an explicitly low-priority slow signal (SRS 2.2.1) that never
  blocks the pipeline regardless of outcome.
- [2026-08-27] Corrected every categorical field's taxonomy in `config/field_sets.yaml`
  against real live-API values (probed at four diverse US points: LA urban, Sierra forest,
  Central Valley agriculture, Napa wine country) rather than the original guessed
  taxonomies: `lcms_class` uses the real USFS LCMS Land_Cover classes ("Trees", "Barren or
  Impervious", ...), `land_use_class` uses the real LCMS Land_Use classes ("Developed",
  "Forest", ...), `fire_hazard_severity_zone_class` uses title-case hyphenated CAL FIRE
  strings ("Non-Wildland", "Very High"), `nearest_road_class`/`nearest_major_road_class`
  use Overture Transportation classes ("secondary", "unclassified", "service", ..., not
  local/collector/arterial), `surface_management_agency` uses "private_or_unknown" (not
  "private"), `primary_building_overture_class` uses real Overture building classes
  ("retail", "hospital", ...), and `mobile_5g_coverage_class` uses the real FCC BDC classes
  ("5g_high_speed", "5g_basic", "5g_marginal"). The policy engine's urban-hotspot guard
  (FR-7/FR-8) was moved from checking `lcms_class == "developed"` to checking
  `land_use_class == "Developed"`, since `land_use_class` is the field whose real value
  literally means "developed/urban" - `lcms_class`'s closest real analogue ("Barren or
  Impervious") is a land-cover, not land-use, concept and is a poorer semantic match.
- [2026-08-27] Unordered categorical one-hot vocabularies (`lcms_class`, `land_use_class`,
  `overture_class`) are fixed, finite category lists taken from the source catalogs' published
  class lists, with an explicit `other` bucket for anything unseen, so the feature vector length
  never depends on what W happens to return in a given call.
- [2026-08-27] V2 LANDFIRE is not the old ArcGIS GPServer `submitJob`. A live probe of
  `https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/GPServer/.../submitJob`
  returned the Next.js HTML form. The machine API is `GET /api/healthCheck`,
  `GET /api/products`, `POST /api/job/submit` (JSON: `Email`, `Layer_List`,
  `Area_of_Interest` as `W S E N` EPSG:4326, `Resample_Resolution`, `Output_Projection`),
  `GET /api/job/status?JobId=`. Submit returns a UUID `jobId`. Status `outputFile` is a zip
  of a multi-band GeoTIFF, dtype int16, nodata -9999, band descriptions like
  `LF2024_FBFM40_CONUS`. Default CRS is a local Albers centered on the AOI; we pass
  `Output_Projection: "4326"`. CH/CBH are stored as m*10, CBD as kg/m3*100. Layer picker
  prefers newest `geoAreas == "All"` (LF2024 fuels, LF2020 Elev/SlpD/Asp). LF2025 FBFM40
  is only SW/NW; seasonal `LF2025_FBFM40_SP26` must not win. Non-burnable FBFM40 codes
  used for AOI clipping: {0, 91, 92, 93, 98, 99}.
- [2026-08-27] ELMFIRE Fortran was not compiled in this environment (no Docker, no
  pre-built `elmfire` binary). Cell2Fire (GPL-3.0) is not used. The licensing seam is
  still real: `spread_service/` is a separate HTTP process; `src/` talks to it only via
  `POST /spread_run` and never imports it (enforced by `tests/test_licensing_seam.py`).
  The process currently runs an original Rothermel-rate + Huygens elliptical raster
  solver using published Scott & Burgan FBFM40 characteristic ROS (not a copy of ELMFIRE
  or Cell2Fire). If `SPREAD_ENGINE_BIN` points at a binary that speaks
  `bin inputs.json outputs.json` with the same output keys, the server execs it instead
  and falls back to the internal solver on failure. Ensemble member 0 is unperturbed;
  members 1..N-1 perturb wind speed/direction and RH.
- [2026-08-27] NIFC Interagency Fire Perimeter History
  (`InterAgencyFirePerimeterHistory_All_Years_View`) is live and queryable, but it is a
  *final* mapped perimeter layer (`FEATURE_CA` typically "Wildfire Final Fire Perimeter";
  `DATE_CUR` is the map date, e.g. `20061102000000`). Native coordinates are not WGS84;
  `outSR=4326` is required (verified: first vertex `-123.23, 47.86`). These polygons are
  not t0 operational snapshots. Using them as E-side `dist_perim_m` at ignition would leak
  the outcome (SRS 6.1). Training still uses the ignition-centroid proxy for t0 distance
  and attaches a delegated `spread_vector` via `--with-spread` instead. IRWINID is often
  null on older historic records.
- [2026-08-27] This V2 run was issued two Mireye keys (not three) and no FIRMS MAP_KEY,
  Slack, or SMTP. Existing degraded paths apply: FIRMS sets `firms_unavailable`; delivery
  is log-only. Two keys round-robin at 300 rpm each on the Growth plan.
- [2026-08-27] H2 (W-conditioned calibration beats raw delegated field on event/HUC/state
  held-out fires) is implemented as `scripts/evaluate_h2.py`.
- [2026-08-28] Path B (`scripts/enrich_spread_vectors.py`) attached `spread_vector` to the
  existing V1 jsonl (4300 samples, 458 events) using LANDFIRE + `spread_run` only — no extra
  Mireye credits. Historic jobs **ignite at the MTBS centroid**; they do **not** seed the
  gold final perimeter (SRS 6.1 leakage). A first 20-row sidecar that did seed the final
  polygon was discarded. LANDFIRE historic tiles are clamped to 0.35° around the centroid
  (live ask/watch still uses the full wind-projected incident AOI). Rasters move over npz,
  not giant JSON, with a 900s HTTP timeout.
- [2026-08-28] Historic Path B weather is a documented Scott & Burgan reference wind
  (2.2 m/s, u-component) so the 7-member ensemble is not degenerate. It is **not**
  HRRR-at-t0. Live V2 still uses HRRR.
- [2026-08-28] H2 on those labels: **validated**. Calibrated PR-AUC 0.865 vs raw engine
  p72 0.637 (ΔAP +0.228); random-W collapse ΔAP +0.079. The comparison is against
  `rothermel_huygens_v1` behind the `spread_run` seam, not a compiled ELMFIRE binary.
  Target remains V1 `y` (in final MTBS perimeter), not a perimeter-time-series arrival
  label (SRS 6.1 V2 gold). Arrival MAE vs a true t0 front is still unevaluable.
- [2026-08-28] A V2 pickle is 5 dims wider than V1. `model_infer` always concatenates the
  spread block when `v2_calibration` is set, using the outside-AOI placeholders
  `[72, 24, 0, 0, 0]` when no incident field exists, so a far-away ask site does not crash.
- [2026-08-28] **R0 timed-perimeter labels.** GeoMAC 2015–2019 + NIFC WFIGS Daily 2020+
  (`WFIGS_Daily_Perimeters_Public`). t0 = first snapshot. Already-inside-seed excluded.
  Never-hit points are negatives only if the series lasts ≥ T hours. On the 4300-row /
  458-event book: 35 events have ≥2 timestamps; **246 rows / 32 events** are evaluable at
  72 h (**27 pos / 219 neg**, 11 events carry every positive). 325 rows were already
  burned at seed. y_24 has 9 pos, y_48 24, y_72 27. Most of the book is n_times=0 (149
  events) or 1 (274). WFIGS Daily contributed 2305 rows but almost all are single-snapshot;
  the multi-time series that actually label arrival are GeoMAC 2019 (34 fires) and one
  GeoMAC 2017 fire. Map methods include Infrared Image and IR Image Interpretation, not
  only sketches. We do **not** fall back to MTBS in/out to inflate N.
- [2026-08-28] **R1 HRRR-seeded operational spread.** `spread_field_for_timed_seed` ignites
  the first operational/IR ring (not the MTBS final scar, not the ignition centroid) and
  drives `spread_run` with HRRR-at-seed. 35 fires got a field, **hrrr_fail=0**, weather
  source `hrrr_at_seed` on every field fire, 20/35 series contain an IR method. All 246
  evaluable_72 rows received `spread_vector_hrrr`. Engine identity remains
  `rothermel_huygens_v1` (7 members, 72 h); `SPREAD_ENGINE_BIN` unset — not compiled
  ELMFIRE. LANDFIRE tiles still clamped to 0.35° around the MTBS centroid.
- [2026-08-28] **R2/R4 full-W GBM on those labels** (`src/model/arrival_eval.py`,
  `data/models/arrival_head_report.json`). X = 200-D W roles A–I (including
  `nearest_fire_perimeter_distance_m`) + E without `dist_perim_m` /
  `wind_ros_ellipse_dist_m` + 5 engine floats. Primary protocol: leave-one-event-out
  pooled PR-AUC (32 folds). Secondary: 20× GroupShuffleSplit (noisy; 3 positives in the
  seed-42 test fold).

  | Estimator | LOGO PR-AUC |
  |---|---|
  | Rank by `-eta_hours` (no model) | **0.422** |
  | Rank by engine p72 | 0.221 |
  | Rank by `-dist_perim_m` (withheld from X) | 0.331 |
  | Rank by `-nearest_fire_perimeter_distance_m` (in W) | 0.110 (chance) |
  | GBM engine-only | 0.262 |
  | GBM W-only | 0.151 |
  | GBM E-nogeom | 0.098 |
  | GBM full W+E+engine | 0.282 |
  | GBM shuffled-W | 0.165 |
  | Prevalence | 0.110 |

  Letter-of-protocol kill test vs p72 **passes** (0.282 > 0.221+0.01 and 0.282−0.165 >
  0.01). Kill test vs the actual engine ranker (`-eta_hours`) **fails** (0.282 < 0.422).
  A GBM on this N is worse than sorting by the engine's ETA. Retrain-ablating all 79
  field groups: **2 help** by >0.01 (`lightning_annual_flash_days` +0.077,
  `near_surface_wind_speed_annual_mean_ms` +0.027), **30 unused** (delta 0, including
  `nearest_fire_perimeter_distance_m`), **36 hurt** (joint GBM overfit; largest:
  `E:wind_speed_ms` −0.107, snow-cover days −0.089, `ENG:eta_hours` −0.083, elevation
  −0.082). Role D is the only role with a clearly positive retrain delta (+0.063); roles
  B/H/G *improve* when dropped. Grouped permutation on the 3-positive GSS test is too
  noisy (eta ±0.14) to override LOGO. The V1 0.96 `-dist_perim_m` scar-ranker does **not**
  transfer to timed arrival (0.331). H2-on-MTBS-y remains a scar classifier, not this
  result.
- [2026-08-28] **Job C / 2023+ Daily tapes / compiled ELMFIRE.** WFIGS Daily 2023–2026
  attribute inventory: 52,393 rows / 24,192 UniqueFireIdentifier fires; **407 usable**
  after CONUS + Daily≥3 + n_times≥4 + span≥72 h + acres 500–2e6 (one 282-million-acre
  GIS explode dropped). Join is exact `attr_UniqueFireIdentifier`, not the 25 km bbox
  nearest-vertex picker that glued MTBS events to 0.1 ac neighbors. ELMFIRE 2025.0212
  was compiled outside the tree (`/home/ubuntu/elmfire`, EPL-2.0, never imported).
  `spread_service/elmfire_bin.py` maps `spread_run` JSON/npz → GeoTIFF + `elmfire.data`
  → TOA. `SPREAD_ENGINE_REQUIRED=1` refuses Huygens fallback. Job C head is a small
  logistic calibrator on `(eta, σ, p_T, W_allowed)`, not a 218-D GBM. W_allowed drops
  NDVI, current LCMS/canopy, burn year, drought, and `nearest_fire_perimeter_distance_m`.
  y is R0 on Daily rings (already-in-seed out; censored if tape < 72 h).
- [2026-08-28] Job C LANDFIRE AOI is the **72 h Daily envelope ∪ seed**, not the
  months-later final ring. Boxing on the last tape ring then capping 0.90° around that
  centroid dropped the seed off-grid (empty PHI → adapter 500). Adapter now writes TOA
  grids as an npz sidecar; nested-list JSON of a 90 m LANDFIRE tile was OOM-adjacent.
  Failed ELMFIRE fires are not written into the book so `--resume` retries them. W
  encoding skips rows without an `elmfire_*` field. Huygens still refused.
- [2026-08-29] Job C W encoder died on HTTP 429 after a `--resume` bursted ~100
  cached-nearby rows in seconds. The in-process limiter is empty after a restart,
  so it cannot see the previous process's rpm. `_pick_key_index` now waits when
  every key is at cap instead of firing a request that will 429. `_request` retries
  429 up to 8 times with 20s–90s backoff (Retry-After honored). `encode_job_c_w.py`
  adds `--startup-pause` / `--batch-pause` and retries rate-limited batches.
- [2026-08-29] First Job C LOGO cut at **25 events / 888 evaluable rows**
  (`elmfire_2025.0212`, Huygens 0). Missing ETA (733 rows never burned in the
  72 h TOA) is imputed to 72 h; missing p72 to 0. Brier: isotonic(eta) 0.180,
  logistic engine 0.180, raw p72 0.223, engine+W_allowed 0.244, shuffled-W 0.201.
  Kill test **failed**. 171-D W_allowed overfits 25 leave-one-event folds;
  this is not a 218-D GBM and not MTBS in/out. Collection continues.
- [2026-08-28] ELMFIRE adapter coarsen used `from_bounds(lon, lat)` on a UTM
  grid, so any LANDFIRE tile wider than 800 cells wrote PHI in degree-space and
  the seed never burned (`no seed cells in phi`). Coarsen now `transform_bounds`
  to UTM metres. Confirmed: `2024-WASES-000173` (previous 500) runs
  `elmfire_2025.0212` after the 72 h AOI + this fix.

