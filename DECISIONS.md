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
