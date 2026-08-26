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
- [2026-08-27] Unordered categorical one-hot vocabularies (`lcms_class`, `land_use_class`,
  `overture_class`) are fixed, finite category lists taken from the source catalogs' published
  class lists, with an explicit `other` bucket for anything unseen, so the feature vector length
  never depends on what W happens to return in a given call.
