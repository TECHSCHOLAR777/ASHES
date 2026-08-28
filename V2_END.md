# V2 Completion Checklist

## Step 0 — verified

`pytest tests/ -q` → **151 passed** on Linux (2026-08-28). Live `ask` at 34.05, -118.24
with cached Mireye W (no extra credits) produced `policy_version: v2.0.0`,
`model_version: h_fire_v20260828T012333Z`, `action: monitor`. FIRMS remains
`firms_unavailable` (no MAP_KEY). Site was far from the nearest WFIGS perimeter, so
`spread_field_version` is null — that is correct, not a miss.

## Step 1 — FR/AC status (2026-08-28)

| Item | Status |
|---|---|
| FR-4/FR-5 AOI | Implemented. Live: wind-projected buffer around active WFIGS perimeter, clipped to burnable FBFM40. Historic Path B: 0.35° clamp around ignition centroid. |
| FR-20/FR-21 spread + LANDFIRE | Implemented. Out-of-process `POST /spread_run`. Solver is original Rothermel+Huygens (`rothermel_huygens_v1`). Compiled ELMFIRE Fortran was never in this tree; `SPREAD_ENGINE_BIN` is the exec adapter. |
| FR-22 ensemble sigma | Implemented (member 0 unperturbed; 1..N-1 perturb wind/RH). |
| FR-26 calibration head | Implemented and **trained** on Path B labels. `h_fire_v20260828T012333Z`, `v2_calibration: true`. |
| FR-38 ETA policy | Implemented (`policy_version: v2.0.0`). |
| FR-56/FR-57 Response Agent | Implemented (AOI water grid + OSM routing). |
| AC-9 out-of-process field | Pass (`test_licensing_seam.py`: `src/` never imports `spread_service`). |
| AC-10 ETA-framed cards | Pass in tests. Live LA point was too far for a usable perimeter sample. |
| AC-11 cost ~ incidents, licensing seam | Seam verified. |
| **AC-12/AC-13 H2** | **`validated`** on Path B labels (see Step 2). |

Polygons were not skipped. V1 geometry is point/optional parcel. V2 live geometry is
incident-first. Training still uses MTBS **final** polygons as gold `y` only; t0 distance
stays the ignition-centroid proxy (SRS 6.1). Historic `spread_run` ignites at that centroid
and does **not** seed the gold perimeter.

Imagery: live EO chips are **out of scope** on the hot path (smoke-blocked). FIRMS thermal
hotspots are E, not chips. AlphaEarth 64-D is a V2 *gated optional* (only if ablation shows
a fuel-texture hole). V1 per-role ablation did not show that hole; AlphaEarth was not
pulled in. Fine-tuned Prithvi / distilled ELMFIRE surrogate remain V3.

## Step 2 — Path B (done, no extra Mireye)

```bash
python scripts/enrich_spread_vectors.py --workers 2 \
  --output data/training/real_conus_2015_2023_with_spread.jsonl
python scripts/enrich_spread_vectors.py --resume --workers 1   # one SSL retry
python scripts/evaluate_h2.py
python scripts/train_model.py
```

| | |
|---|---|
| Input | `data/training/real_conus_2015_2023.jsonl` (4300 rows / 458 events, W/E already paid) |
| Output | `data/training/real_conus_2015_2023_with_spread.jsonl` (4300 rows, all with `spread_vector`) |
| Failed fires | 1 transient LFPS SSL error, recovered on `--resume` |
| `h2_status` | **validated** |
| Calibrated PR-AUC | 0.865 |
| Raw engine p72 PR-AUC | 0.637 |
| ΔAP vs raw | +0.228 |
| Random-W ΔAP | +0.079 |
| Brier | 0.119 |
| Model | `h_fire_v20260828T012333Z` |

Honest limits of H2 here: reconstructed `coords_source=reconstructed_centroid_circle`;
historic wind is 2.2 m/s reference (not HRRR-at-t0); engine is `rothermel_huygens_v1`;
gold `y` is still in-final-perimeter, not a perimeter-time-series arrival time.

## Step 3 — docs / tests

- README H2 paragraph replaced `unevaluable_no_real_spread_labels` with the validated numbers.
- HANDOFF / DECISIONS updated.
- Full `pytest tests/ -q` re-run after the V2 `model_infer` pad-missing-spread fix.

## Do not

- Re-collect via Path A (`build_training_set.py --with-spread`) unless you mean to spend
  Mireye credits again.
- Treat `data/cache/w_cache.db` as spread labels.
- Seed historic `spread_run` with the MTBS final perimeter.
- Commit `.env`, jsonl, pickles, LANDFIRE caches, or API keys.
