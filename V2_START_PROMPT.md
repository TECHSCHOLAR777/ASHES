# V2 Build Prompt — Wildfire Site-Event Copilot

You are an autonomous senior software engineer picking up V2 of an existing, working
production system. V1 is complete, tested, and pushed to
`https://github.com/TECHSCHOLAR777/ASHES`. Do not rebuild V1. Extend it.

## Do not wait for the training data job to start

A large real MTBS-labeled training-data collection job (`scripts/build_training_set.py`)
may still be running against the live Mireye/FIRMS/HRRR APIs when you start. **You do not
need it to finish before beginning V2 engineering.** Almost none of the V2 build depends on
it: the LANDFIRE client, incident-first AOI geometry, the out-of-process `spread_run`/ELMFIRE
seam, the ensemble-sigma mechanism, the ETA-keyed policy extension, and the Response Agent
V2 upgrades (AOI water map, OSM routing) are all independent of the training dataset. Build
all of that in parallel, starting immediately.

The one thing that genuinely needs the finished (or a frozen snapshot of the) dataset is the
final step: retraining `h_fire`'s W-conditioned calibration head on real data and running the
H1/H2 held-out evaluation (SRS 6.5/6.6) - that evaluation's credibility depends on having
enough real event/HUC/state-diverse fires, which is the entire reason that job collects as
much real data as it does. Check `data/training/*.jsonl` and
`python scripts/credit_report.py` to see current volume; if the collection job has already
finished by the time you read this, just use the file as-is - no need to wait further.

## Before writing any code

1. Clone the repo and run `pytest tests/ -v`. All 111 tests must pass. If they do not, stop
   and fix the environment before doing anything else - do not build on a broken baseline.
2. Read `HANDOFF.md` in full. It explains exactly what V1 is, where everything lives, and
   what V2 needs to build.
3. Read `SRS_fire.md` in full, especially section 8.2 (V2 end-to-end), 8.3 (comparison
   table), 5.5/5.6 (spread + model FRs), 6.4-6.6 (model stages, evaluation, research claim),
   and 2.6.1/7.7 (the ELMFIRE licensing boundary - this is a hard constraint, not a
   suggestion).
4. Read `DECISIONS.md` in full. It is the record of every place the SRS's assumptions did
   not match the real APIs during the V1 build, and exactly how each was fixed with live
   evidence. The Mireye field taxonomies, MTBS's coordinate system, HRRR's grid projection,
   FIRMS's real quota, and several other things are NOT what you would guess from
   documentation alone - this file saves you from re-discovering all of it by hand.
5. Confirm the live pipeline actually works before changing anything:
   `python scripts/ask.py --lat 34.05 --lng -118.24 --q "test"` should produce a real
   ActionCard with citations from live Mireye/FIRMS/WFIGS/HRRR/NWS calls.
6. Check `data/training/*.jsonl` for whatever real MTBS-labeled training data exists from
   the V1 handoff and run `python scripts/train_model.py` if it hasn't been trained on yet -
   V2 should start from a real-data baseline, not the synthetic bootstrap model.

## What you are building

V2 makes fire spread the spine of the system without ever learning the underlying physics
from scratch (SRS explicitly forbids this - the public next-day-spread SOTA ceiling, WSTS+
Res18-UNet at ~0.47 AP, is the documented reason to delegate rather than learn). Concretely,
in dependency order:

1. **LANDFIRE client** (`src/clients/landfire.py`): fetch FBFM40 fuel, canopy (CBD/CBH/CC/CH),
   and DEM raster tiles for an incident AOI. Verify the real API/download shape against a
   live call before writing the parsing code - do not guess field names or response
   structure the way an earlier pass on this project initially guessed at Mireye's and
   MTBS's shapes and had to fix them against live evidence (see `DECISIONS.md` for what that
   looked like and why it matters).
2. **Incident-first AOI geometry** (FR-4, FR-5): a wind-projected buffer around an active
   WFIGS perimeter (reuse `src/clients/wfigs.py`), clipped to burnable fuel from LANDFIRE.
   Extend the `{mode, geom, cells[], source_layer, vintage}` geometry contract already
   described in SRS FR-5; do not invent a parallel contract.
3. **The `spread_run` licensing seam**: stand up ELMFIRE (EPL-2.0) as a genuinely
   out-of-process service (its own container/process, called over an internal API or queue -
   never imported as a Python library into this codebase). This is required by SRS 2.6.1/7.7,
   not optional hardening. Cell2Fire (GPL-3.0) is the fallback only if ELMFIRE truly cannot
   be stood up, and even then must be isolated the same way with a legal review flagged in
   `DECISIONS.md`.
4. **Ensemble runs for sigma** (FR-22): wind/moisture perturbations sized to produce a
   genuine per-site arrival-time uncertainty, not a placeholder constant.
5. **The W-conditioned calibration head** (FR-26): extend `src/model/h_fire.py`'s
   `model_infer` contract - it must remain the same function signature and stable output
   keys the V1 agent already depends on (`y_hat`/`sigma`/`baseline_y`/`model_version`, now
   adding `eta_hours`/`eta_sigma_hours`/`p_burn_by_T`/`spread_field_version`). Do not change
   the agent to accommodate a different contract; change the model to fit the existing one.
6. **ETA-keyed policy** (FR-38): extend `src/policy/engine.py`'s table (do not replace it -
   it is a versioned, inspectable table by design). Keep the existing high-sigma-suppresses-
   evacuate safety rule; add the equivalent for high `eta_sigma_hours`.
7. **Response Agent V2 upgrades** (FR-56/FR-57): dense AOI water grid, OSM road-network
   routing replacing the V1 distance/speed proxy in `src/agents/response_agent.py`.
8. **Real training data at V2 scale**: extend `scripts/build_training_set.py` /
   `src/model/training_data.py`'s pattern to pull real ELMFIRE ensemble outputs and
   real WFIGS perimeter time series (removing the V1 ignition-centroid proxy documented in
   `training_data.py`'s module docstring) as the true `dist_perim_m`/arrival-time labels.
9. **The falsification gate** (SRS 6.6, AC-12/AC-13): the actual research deliverable is
   proving (or honestly disproving) that W-conditioned calibration beats raw ELMFIRE on
   event/HUC/state held-out fires, with a random-W-shuffle collapse probe as a gate - the
   same pattern `src/model/train.py` already implements for V1's H1 claim. Do not skip this
   to ship faster; an honest "H2 falsified, shipping the delegated model alone" is an
   acceptable and expected possible outcome per the SRS itself (6.6).

## Non-negotiables (carried over from V1, still apply)

- No stubs, no `TODO`, no `NotImplementedError` in `src/` or `scripts/`.
- No invented numbers anywhere in an ActionCard/ResponseCard - every number traces to a
  real API response or a deterministic computation in code.
- Verify every new external API's real request/response shape with a live call before
  writing the client code against assumed documentation. This project's V1 build hit
  real, live-verified surprises in Mireye, MTBS, FIRMS, and HRRR that no amount of reading
  docs alone would have caught - assume LANDFIRE and ELMFIRE will have their own.
- Write tests for every new module (mock the live calls; keep the pattern already used
  throughout `tests/`).
- Commit in small, honest, professional increments as you go - no AI-slop language, no em
  dashes, no giant single commits. Look at the existing 37 commits on `master` for the tone
  and granularity expected.
- Update `DECISIONS.md` for every place a real API's behavior differs from what the SRS or
  its own documentation implied - that file is the project's memory; keep it accurate.

## When you are done

Run the same acceptance criteria V1 was held to, extended for V2 (SRS section 12.3/12.4,
AC-9 through AC-13): ELMFIRE running out-of-process with a versioned field, ETA-framed cards
with high-sigma suppression, cost scaling with active incidents not site count, and an honest
pass/fail report on the H2 research claim. Report which V2 FRs are fully implemented, which
V2 ACs pass, and the real H2 result - validated or honestly falsified - the same way V1's
final report covered H1.
