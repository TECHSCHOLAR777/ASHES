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
- [2026-08-27] Unordered categorical one-hot vocabularies (`lcms_class`, `land_use_class`,
  `overture_class`) are fixed, finite category lists taken from the source catalogs' published
  class lists, with an explicit `other` bucket for anything unseen, so the feature vector length
  never depends on what W happens to return in a given call.
