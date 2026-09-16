# V1 → V2 Handoff

## Job C (2026-08-28, in progress)

Showcase is **not** “a GBM ranks 72 h arrival.” It is:

`P(y_72 | ELMFIRE eta, σ, p_T, W_allowed)` vs `P(y_72 | eta, σ, p_T)` vs raw `p_T`,
event-held-out, Brier / log-loss / reliability, shuffle-W kill. Same ETA, different
parcel, different hit probability.

**Book:** WFIGS Daily 2023–2026, exact UniqueFireIdentifier. Inventory:
`data/training/job_c_2023/tape_inventory.json` (407 usable CONUS fires; 404 tapes on
disk, 3 empty IDs). After all tapes: **7987 rows / 283 events / 2099 positives /
7947 evaluable_72** (`book_summary.json`). Points are the 72 h growth annulus, not
the final ring. LANDFIRE AOI is that same 72 h envelope ∪ seed. ELMFIRE: compile
https://github.com/lautenberger/elmfire branch `2025.0212` *outside* this repo, then

```
export SPREAD_ENGINE_BIN=$PWD/spread_service/elmfire_bin.py
export SPREAD_ENGINE_REQUIRED=1
export ELMFIRE_BIN=/home/ubuntu/elmfire/build/linux/bin/elmfire
export SPREAD_SERVICE_PORT=8766
python scripts/enrich_job_c_elmfire.py --resume
```

Huygens is not the claim. If the binary is missing the Job C script fails closed.

**W:** new Mireye spend on new coordinates (`scripts/encode_job_c_w.py --resume --batch-size 10`).
Fetch A–I; the head subsets W_allowed (`src/model/w_allowed.py`). Snapshot
`data/training/job_c_2023/progress.json`. Encoder waits on Mireye 429 (in-process
limiter now blocks when both keys are at cap; `--startup-pause` after a restart).
If W exits after its pending list, `--resume` again against the growing elmfire
book. Huygens rows are skipped. Failed ELMFIRE fires are not written so
`--resume` retries them (empty PHI before the coarsen fix; a few Fortran runs
with no TOA).

**Eval (LOGO, 283 events / 7947 rows, `elmfire_2025.0212` only, 0 Huygens).**
The 171-D W_allowed dump never had a top-k (the head is logistic, not a GBM).
Within-fire Spearman vs the engine residual + FDR: **no field survived**. Exploratory
top-8 still used for the same LOGO protocol. Map: `data/models/job_c_w_correlation.json`.

| Head | Brier | log-loss | PR-AUC (side) |
|---|---|---|---|
| Raw engine p72 | 0.247 | 3.411 | 0.416 |
| Isotonic(eta) | 0.163 | 0.504 | 0.486 |
| Logistic P(y\|engine) | **0.163** | **0.503** | 0.465 |
| Logistic P(y\|engine, top-8 W) | 0.168 | 0.542 | 0.447 |
| Logistic shuffled top-8 W | 0.164 | 0.506 | 0.462 |

Kill still **failed**. FDR selected **nothing** (within-fire perm p ≈ 1: the tiny |ρ| is between-fire, not parcel). Top-8 ΔBrier **−0.005**. Shuffled W still beats real W. 5693/7947 rows never received a 72 h TOA.

**GBM calibrator (same task, 283 / 7947, Huygens 0):** HistGradientBoosting
`predict_proba`, LOGO Brier, OOS permute-each-field ablation. Kill still **failed**.
No field earned keep (threshold ΔBrier 0.001). Report: `data/models/job_c_gbm_report.json`.

| Head | Brier | log-loss |
|---|---|---|
| Raw engine p72 | 0.247 | 3.411 |
| Isotonic(eta) | **0.163** | **0.504** |
| GBM P(y\|engine) | 0.164 | 0.505 |
| GBM P(y\|engine, W_allowed 171-D) | 0.166 | 0.513 |
| GBM shuffled W | 0.164 | 0.507 |

Closest-to-useful ablation (still below keep): surface management +0.0005, road length +0.0003. Climate (temp, snow) **hurts** when left in. Switching logistic → GBM did not create a Mireye parcel effect.

Do not re-run Path A / Path B / R0–R4 on the 2015–2022 MTBS book for this claim.

---

## V2 status (2026-08-28)

V2 Path B is closed. `scripts/enrich_spread_vectors.py` labeled all **4,300** V1 samples
across **458** MTBS events with a delegated `spread_vector` and **no extra Mireye credits**.
`python scripts/evaluate_h2.py` reports **`h2_status: validated`**:

| Metric | Value |
|---|---|
| Samples / events | 4300 / 458 |
| Calibrated PR-AUC | 0.865 |
| Raw engine p72 PR-AUC | 0.637 |
| Calibrated − raw ΔAP | +0.228 |
| Random-W collapse ΔAP | +0.079 |
| Brier (calibrated) | 0.119 |
| Model | `h_fire_v20260828T012333Z` (`v2_calibration: true`) |

Honest limits (also in `DECISIONS.md` / `V2_END.md`): ignition-centroid seed (not gold
final perimeter), 0.35° historic AOI clamp, 2.2 m/s reference wind, Rothermel-Huygens not
compiled ELMFIRE, reconstructed sample coordinates. Live `ask` still uses the full
wind-projected WFIGS AOI + HRRR.

**R0–R4 (timed arrival, 2026-08-28):** those H2 numbers are on V1 `y` = inside the final
MTBS scar. Ranking by `-dist_perim_m` already scores ~0.96 there. Timed GeoMAC/WFIGS Daily
labels give **246 evaluable rows / 32 events / 27 positives**. R1 re-seeded `spread_run`
from the first operational ring with HRRR-at-seed (35 fires, hrrr_fail=0). Full-W leave-one-
event-out GBM PR-AUC **0.282** beats p72 rank (0.221) and shuffled W (0.165) but **loses to
ranking by `-eta_hours` (0.422)**. Only lightning-flash-days and near-surface wind
climatology help by >0.01 when every role and field is retrain-ablated.
`data/models/arrival_head_report.json`.

Do not re-run Path A (`build_training_set.py --with-spread`) unless you intend to spend
Mireye credits. Resume Path B with `--resume` if a sidecar already exists.

---

## Where things stand


V1 (the wildfire site-event copilot) is complete and pushed to
`https://github.com/TECHSCHOLAR777/ASHES` as of commit `aca8b16` (37 commits on
`master`). Every V1 functional requirement in `SRS_fire.md` (FR-1 through FR-55) is
implemented, tested, and has been validated against the *real* live APIs, not just mocks:
Mireye, NASA FIRMS, WFIGS, NWS, HRRR, USGS, and MTBS. 111 automated tests pass
(`pytest tests/ -v`). The watch loop has completed real multi-cycle polling runs against 5
live sites; `ask` mode and the Response Agent have produced real ActionCard/ResponseCard
output end to end.

Read `DECISIONS.md` before touching anything - it is a running log of every place this build
found the SRS's assumptions did not match the real APIs (field taxonomies, CRS handling,
rate limits, scheduler behavior) and how each was fixed, with the live evidence that proved
it. Skipping it means re-discovering the same bugs by hand.

**Update (2026-08-27): the real training data collection finished and H1 is validated.**
`data/training/real_conus_2015_2023.jsonl` holds 4,300 real samples across 458 distinct
MTBS fires (2015-2023, CONUS-wide, ~262,300 real Mireye credits spent). `h_fire` has been
retrained on it (`h_fire_v20260827T181439Z.pkl`, not committed - gitignored, regenerate with
`python scripts/train_model.py` if needed): **PR-AUC 0.7414, Brier 0.1666, and the random-W
collapse probe drops PR-AUC to 0.6382 when W is shuffled - a real 0.103 AP loss.** That is
the SRS's own H1 gate (AC-7): cited Mireye W measurably improves the score beyond E alone,
on real held-out fires. H1 is validated, not just plumbing-tested. V2 should treat this as
the baseline `h_fire` (Variant A) result to beat with the W-conditioned calibration head.

## What V1 actually is

An unattended, cited copilot for a book of named US sites: watches live fire signals
(NWS Red Flag, NASA FIRMS hotspots, WFIGS incidents/perimeters, HRRR weather), enriches each
site with cited Mireye physics (fuel, terrain, hazard, access, water, ~61 fields), scores it
with a small trained model, and applies a deterministic policy table to produce one of six
actions (`no_action`, `monitor`, `prepare`, `protect_asset`, `evacuate_site`,
`inspect_after`). On `protect_asset`/`evacuate_site` a second Response Support Agent produces
a tactical dossier (water sources, access routes, hazmat, agency, comms) with zero
LLM-invented numbers - an automated validator rejects any brief that introduces a number not
already on the card.

## Project layout (start here)

- `SRS_fire.md` - the engineering contract. Read this fully before writing any V2 code.
- `DECISIONS.md` - every real-world correction made during the V1 build. Required reading.
- `README.md` - install, quick start, architecture diagram, known limitations.
- `src/clients/` - one file per external API (`mireye.py`, `firms.py`, `wfigs.py`, `nws.py`,
  `hrrr.py`, `usgs.py`, `mtbs.py`). All are real, tested, and rate-limit-aware.
- `src/features/` - `w_encoder.py` (typed/masked Mireye encoding), `e_packer.py` (live E
  feature packing), `ros_ellipse.py` (the V1 crude wind-ellipse spread proxy V2 replaces).
- `src/model/` - `h_fire.py` (the stable `model_infer` contract V2 must keep), `train.py`
  (CPU training pipeline), `training_data.py` (real MTBS-labeled sample construction).
- `src/agents/` - `main_agent.py` (the fixed-order pipeline), `response_agent.py`,
  `watch_runner.py` (scheduling/dedup).
- `src/policy/engine.py` - the deterministic policy table. V2 adds ETA-keyed rules here
  (FR-38) without touching the agent.
- `config/field_sets.yaml` - the Mireye field catalog with real, live-verified taxonomies.
- `scripts/build_training_set.py` - real MTBS-labeled training data collector, with adaptive
  concurrency control and resume support. This is the pattern to extend for V2's richer
  training needs (LANDFIRE tiles, ELMFIRE ensemble runs).

## What V2 actually needs to build (from SRS section 8.2)

V1's `model_infer` and ActionCard contracts are **stable across V1 and V2** by design - V2
changes what happens *behind* the contract, not the agent or the schema shape. Concretely,
per the SRS:

1. **Incident-first AOI geometry** (FR-4): a wind-projected buffer around an active WFIGS
   perimeter, clipped to burnable fuel, replacing V1's point/parcel-only geometry.
2. **LANDFIRE integration** (FR-21): FBFM40 fuel, canopy (CBD/CBH/CC/CH), DEM as raster
   inputs to the spread engine - never fed to Mireye's role, never used as a Rothermel model
   substitute for Mireye's own fields.
3. **Delegated spread engine, ELMFIRE** (FR-20, licensing constraint in SRS 2.6.1/7.7): run
   **out-of-process** behind a `spread_run` internal API so EPL-2.0 code never links into
   this proprietary agent/model code. This is a hard licensing requirement, not a style
   choice - read SRS 2.6.1 before picking an integration approach.
4. **Ensemble runs for sigma** (FR-22): wind/moisture perturbations to produce per-site
   arrival-time uncertainty.
5. **W-conditioned calibration head** (FR-26): `h_fire` becomes
   `h_fire(raw_field_at_site, Mireye W, E) -> calibrated ETA, P(burn<=T), sigma`, replacing
   V1's plain MLP - but through the *same* `model_infer` contract in `src/model/h_fire.py`.
6. **ETA-keyed policy** (FR-38): extend `src/policy/engine.py`'s table with ETA thresholds,
   keeping the existing sigma-suppression-of-evacuate safety rule (NFR-17).
7. **Response Agent upgrades** (FR-56/FR-57): AOI water map (dense Mireye grid over the
   incident AOI) and OSM road-network routing to replace the V1 distance-proxy ETA.
8. **Research claim H2** (SRS 6.6): W-conditioned calibration must beat *raw ELMFIRE* on
   event/HUC/state held-out fires, with the same random-W collapse gate V1 used for H1. This
   is the actual point of V2 - do not skip the falsification probes to save time.

**Do not build**: a learned-from-scratch spread model (SRS explicitly rejects this, backed
by the WSTS+ SOTA ceiling in 6.2), Cell2Fire as the primary engine (GPL-3.0, use ELMFIRE),
or any live EO chip on the hot path (smoke-blocked, too slow - see SRS 6.3).

## Honest state of the real training data

V1 shipped with a synthetic-bootstrap `h_fire` model (random labels, 500 samples) purely so
the pipeline was runnable end to end before real data existed - this was intentional per the
SRS itself, not a shortcut. A real MTBS-labeled dataset was being collected at handoff time
(`scripts/build_training_set.py`, targeting real Mireye/FIRMS-archive/HRRR-archive data
across thousands of real fire samples). Read `src/model/training_data.py`'s module docstring
for the one honest limitation in that pipeline: `dist_perim_m` is proxied off the fire's
ignition centroid, not a true t0 perimeter, because MTBS publishes only ignition date +
final perimeter, not a time series. Timed GeoMAC/WFIGS Daily series now exist for 35 of 458
events (`scripts/label_arrival_times.py`); 246 sites are evaluable at 72 h. That is the
arrival label. It is not a substitute for a compiled ELMFIRE field.

## Getting started

```bash
git clone https://github.com/TECHSCHOLAR777/ASHES.git
cd ASHES
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your own Mireye/OpenAI/FIRMS keys
pytest tests/ -v       # confirm all 111 tests pass before changing anything
python scripts/ask.py --lat 34.05 --lng -118.24 --q "test"   # confirm live pipeline works
```

Then read `SRS_fire.md` section 8.2/8.3 in full, and start with the licensing seam
(`spread_run` as an out-of-process API) since every other V2 piece depends on that boundary
existing first.
