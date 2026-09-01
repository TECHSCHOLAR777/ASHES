# ASHES: Wildfire Site-Event Copilot

ASHES is a safety-bounded decision-support system for organisations that operate a known portfolio of US sites in wildfire-exposed areas. During an active or imminent fire, it turns fragmented live signals, site-specific environmental context, and spread modelling into a cited, auditable **ActionCard** for each site.

It is built for facilities, operations, and risk teams deciding what to do in the next hours, not for public emergency broadcasting. It does not replace incident command, NWS alerts, or evacuation orders. In particular, it never suppresses an official warning and never represents a modelled fire perimeter as official information.

## The idea

Wildfire response is usually a manual synthesis problem: a team watches alerts, thermal hotspots, incident data, weather, maps, and site details, then makes a time-sensitive call. ASHES automates the evidence assembly while keeping the decision process inspectable.

For every configured site, ASHES can:

- Watch for fire-weather alerts, hotspots, incidents, operational perimeters, and forecast weather.
- Enrich the site with cited information about fuels, terrain, historical fire context, buildings, egress, hazards, water, and governance.
- Estimate fire arrival and probability of burning within 24, 48, and 72 hours when an incident AOI and spread field are available.
- Produce one closed recommendation: `no_action`, `monitor`, `prepare`, `protect_asset`, `evacuate_site`, or `inspect_after`.
- Generate a deterministic response dossier when protection or evacuation is warranted: water sources, access routes, fire-station ETA, hazmat priorities, environmental constraints, and responsible agency.

The output is designed to be useful under pressure: it includes supporting signals, source/vintage metadata, model and policy versions, degraded-data flags, and a short validated brief.

## Why it matters

ASHES reduces the time between a changing fire picture and a defensible operational response. Its value is not a claim to predict every fire perfectly; it is the combination of timely evidence, consistent decision rules, uncertainty disclosure, and a reproducible record of what the system knew at the time.

- **Faster triage across many locations.** A scheduled watch loop evaluates a configured book of sites rather than requiring someone to inspect maps site by site.
- **More grounded decisions.** Live conditions are joined with local fuels, terrain, access, and exposure rather than relying on distance alone.
- **Clear escalation support.** When a site reaches a protection or evacuation state, a separate structured dossier is produced rather than an ungrounded chat response.
- **Auditability.** Every external call, policy decision, model invocation, and quoted credit cost is written to append-only JSONL logs. Cards carry versions and source information.
- **Honest uncertainty.** Missing feeds, stale weather, unofficial perimeters, coarse AOIs, and uncertain arrival estimates are surfaced as flags, not silently filled with plausible values.

## Safety and decision boundaries

| Concern | ASHES behaviour |
| --- | --- |
| Final action | A versioned rule table in `config/policy.yaml` selects the action. The LLM cannot select or override it. |
| Language model | Used only for a prose brief and optional tool orchestration. The brief validator rejects new numbers or an action/severity escalation and falls back to deterministic prose. |
| Official information | NWS alerts are retained; WFIGS perimeters are labelled operational/unofficial. A simulated spread field is never presented as an official perimeter. |
| Failure behaviour | External-service failures degrade the card with flags instead of fabricating data or crashing the watch loop. |
| Scope | This is an internal tool for named commercial or industrial sites, not a consumer “will my house burn?” product or a substitute for emergency management. |

## Architecture

```text
                        CONFIGURED SITE BOOK
                     config/sites.yaml + policy.yaml
                                  |
                                  v
                     +-------------------------+
                     | Watch loop / Ask API/UI  |
                     | APScheduler or FastAPI   |
                     +------------+------------+
                                  |
             +--------------------+---------------------+
             |                                          |
             v                                          v
   LIVE EVENT SIGNALS (E)                       SITE CONTEXT (W)
   NWS / SPC fire weather                       Mireye Fire Intelligence Set
   NASA FIRMS thermal hotspots                  fuels, terrain, hazard, exposure,
   WFIGS incidents + perimeters                 egress, water and governance
   HRRR weather                                 role-aware SQLite cache + vintages
             |                                          |
             +-------------------+----------------------+
                                 v
                    Feature construction and geometry
              E packer · typed W encoder · ROS ellipse
              incident-first wind-aware AOI construction
                                 |
                 +---------------+----------------+
                 |                                |
                 v                                v
       LANDFIRE raster stack              V1 / V2 probability head
       fuels + canopy + terrain           calibrated `h_fire` model
                 |                        baseline + uncertainty
                 v                                |
   +-------------------------------------------------------------+
   |  Out-of-process `spread_run` service                        |
   |  default: Rothermel + Huygens ensemble                      |
   |  optional: compiled ELMFIRE adapter through JSON/NPZ seam   |
   |  outputs: ETA, ETA sigma, P(burn by 24/48/72 h), field id   |
   +-------------------------------+-----------------------------+
                                   |
                                   v
                 Versioned policy engine (the action authority)
                                   |
                     +-------------+--------------+
                     |                            |
                     v                            v
               cited ActionCard             escalation only
               validated prose brief              |
                     |                            v
                     |                 Response Support Agent
                     |                 USGS + OSM + Mireye role J
                     |                 deterministic ResponseCard
                     v                            |
          Slack / email delivery or log-only <-----+
          UI / API / append-only audit log
```

### Runtime flow

The standard prediction path fetches NWS alerts first, then obtains FIRMS, WFIGS incident/perimeter, and HRRR data concurrently. It packs those live features, retrieves only uncached Mireye roles A–I, encodes the combined inputs, and optionally requests an incident-centered spread field. The model produces a calibrated score and uncertainty; the policy engine applies distance, weather, ETA, density, egress, containment, and uncertainty gates to choose the action. The ActionCard is then delivered and persisted.

For `protect_asset` and `evacuate_site`, a separate Response Support Agent enriches the result with operational context. It is deterministic and does not use a trained model to rank or invent resources.

## Technical design

### Evidence layers

ASHES separates information by provenance and cadence:

- **E, event-time data:** NWS CAP alerts and SPC outlook, FIRMS hotspots, WFIGS incident metadata and operational perimeters, and HRRR weather.
- **W, site-world data:** a typed Mireye field catalogue grouped into roles A to J: fuel/vegetation, terrain, fire-weather climatology, hazard priors, ignition infrastructure, exposure, access, compounding hazard, water/governance, and response resources.
- **Spread layer:** an incident-first AOI pulls LANDFIRE fuel, canopy, and terrain rasters. A 7-member wind/moisture ensemble returns a spatial arrival field, arrival-time uncertainty, and probability layers.

W fields are cached with role-dependent freshness windows, type encoded with missing-value masks and confidence features, and tagged with source/vintage information. Join keys and identifiers stay out of the model feature vector. Historic modelling excludes fields that would leak post-event knowledge or use today’s dynamic condition for an older fire.

### Spread-engine seam

The product deliberately keeps wildfire-spread execution outside `src/`. The application communicates with a local HTTP `spread_run` contract; the service can use the built-in Rothermel–Huygens solver or invoke a separately compiled ELMFIRE binary through a JSON/NPZ adapter. This is both an engineering and licensing boundary: the main application neither imports nor bundles the external simulator.

The fallback engine is identified as `rothermel_huygens_v1` in its outputs. ELMFIRE is optional, must be compiled outside this repository, and is never claimed merely because the adapter exists. Set `SPREAD_ENGINE_REQUIRED=1` in a validation run to fail rather than silently use the fallback.

### Model and evaluation

`h_fire` has a stable inference contract so the underlying predictor can evolve without changing cards or policy. The current training path uses a scikit-learn MLP with isotonic calibration, event-held-out evaluation, PR-AUC/Brier reporting, and a random-W collapse probe. V2 conditions the probability head on raw spread outputs; it does not replace the engine’s ETA or probability layers.

The repository includes two historical workflows:

- **MTBS final-perimeter corpus:** useful for building an initial site-risk training set, with explicit safeguards against outcome leakage.
- **Timed-perimeter arrival corpus (Job C):** uses GeoMAC and WFIGS Daily time series to label arrival by horizon and evaluates an event-held-out full-W gradient-boosting head. This is the more appropriate target for validating arrival claims.

Evaluation scripts are falsification gates. If real, suitable labels are absent, they report an unevaluable outcome rather than a pass. The shipped fallback model makes the pipeline runnable, but it is not evidence of operational predictive skill.

### Trustworthy output contracts

Pydantic schemas define the closed ActionCard and ResponseCard. They preserve nulls where data is unknown, for example, a missing incident acreage or a Mireye waterbody name without a verified distance. The brief validator permits only numbers already present in the grounded card and rejects impermissible escalation language. These constraints make it harder for a presentation layer to hide uncertainty or invent operational facts.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/agents/` | Watch, ask, agentic orchestration, response support, place parsing, and action playbooks. |
| `src/clients/` | Source-specific clients for NWS, FIRMS, WFIGS, HRRR, Mireye, LANDFIRE, USGS, OSM, MTBS, and timed perimeters. |
| `src/features/` | Typed W encoding, live E packing, and the V1 wind-projected ROS ellipse. |
| `src/spread/` and `spread_service/` | Spread-service client plus the isolated Rothermel–Huygens and optional ELMFIRE execution layer. |
| `src/model/` | Training, inference, label construction, vintage-safe feature selection, and evaluation. |
| `src/policy/` | Versioned, deterministic action policy. |
| `src/schemas/`, `src/validator/` | Card contracts and grounded-brief validation. |
| `src/serve/` | FastAPI endpoints and the card/map user interface. |
| `config/` | Site book, policy thresholds, showcase scenario, and field catalogue. |
| `scripts/` | Operations, data collection, training, evaluation, and service entry points. |
| `tests/` | Unit and contract tests covering policy, sources, geometry, models, spread results, and agent boundaries. |
| `SRS_fire.md` | Detailed software requirements and acceptance criteria. |
| `DECISIONS.md` | Recorded engineering decisions and evidence behind them. |

## Getting started

### Prerequisites

- Python 3.11 or later (the project was built and tested with Python 3.12.13).
- At least one Mireye API key; two or three keys support round-robin use.
- A NASA FIRMS `MAP_KEY` for hotspot data. Without it, the card degrades with a visible flag.
- An OpenAI API key for the optional tool-calling experience and copy-only brief. Without it, deterministic fallback briefs still work.
- Optional Slack bot and SMTP configuration for actual delivery. Unconfigured delivery uses log-only mode.

### Install

```bash
git clone <repository-url>
cd ASHES
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env  # macOS/Linux: cp .env.example .env
```

Add the relevant credentials to `.env`. Keep `.env` private; it is ignored by Git.

### Run a one-off assessment

```bash
python scripts/ask.py --lat 33.9806 --lng -117.3755 --q "Is this site at risk?"
```

This runs the deterministic data-and-policy path once and prints the ActionCard. It also prints a ResponseCard if the action escalates to `protect_asset` or `evacuate_site`.

For the tool-calling experience, which can invoke the same bounded tools while policy retains decision authority:

```bash
python scripts/ask.py --agentic --lat 33.9806 --lng -117.3755 --q "Is this site at risk?"
python scripts/ask.py --agentic --simulate --lat 33.9806 --lng -117.3755 --q "Simulate ignition here"
```

### Start continuous monitoring and the UI

```bash
python scripts/watch_loop.py
python scripts/serve.py
```

The UI is available at `http://127.0.0.1:8080` by default. Configure sites in `config/sites.yaml`; set thresholds, cadence, AOI size, ensemble parameters, and delivery behaviour in `config/policy.yaml`.

### Test

```bash
pytest
```

## Data, models, and operations

### Bootstrap versus real training

To exercise the full pipeline without an existing corpus:

```bash
python scripts/generate_synthetic_training_data.py
python scripts/train_model.py
```

The synthetic dataset is only a bootstrapping mechanism. It does not establish real-world skill and should not be used to make performance claims.

To collect a real MTBS-backed corpus and train an initial head:

```bash
python scripts/build_training_set.py \
  --bbox -125 24 -66 49 \
  --year-start 2010 --year-end 2023 \
  --min-acres 1000 --max-fires 1000 \
  --positives-per-fire 3 --hard-negatives-per-fire 3 --easy-negatives-per-fire 2 \
  --credit-target 290000 \
  --output data/training/real_conus_2010_2023.jsonl

python scripts/train_model.py
python scripts/evaluate_h2.py
```

To build and evaluate the stronger timed-perimeter arrival workflow:

```bash
python scripts/run_job_c.py --stage inventory
python scripts/run_job_c.py --stage tapes --resume
python scripts/run_job_c.py --stage points --resume
python scripts/run_job_c.py --stage elmfire --resume
python scripts/run_job_c.py --stage w --resume
python scripts/run_job_c.py --stage eval
```

The Job C workflow is intentionally staged so expensive external data acquisition can resume safely. Review the generated report before treating a model change as an improvement.

### Audit trail and local state

- `data/logs/YYYY-MM-DD.jsonl` records tool calls, model calls, policy decisions, delivery attempts, and credit quotes without logging raw keys.
- `data/cache/w_cache.db` stores role-aware Mireye cache entries.
- `data/cache/site_state.db` stores polling and delivery state for idempotency and deduplication.
- `data/models/` stores local model artifacts and evaluation reports.

Check observed Mireye credit use with:

```bash
python scripts/credit_report.py --days 30
```

## Known limitations and honest interpretation

- The default fallback spread solver is a simplified Rothermel–Huygens ensemble, not a guarantee of physical accuracy. Its output must be interpreted as a decision-support signal with disclosed uncertainty.
- Compiled ELMFIRE is an external optional dependency. The adapter is present, but a real binary and required runtime configuration are necessary to use it.
- Operational WFIGS perimeter data can be incomplete or unofficial; ASHES labels it accordingly.
- The SPC Day-1 fire-weather page is parsed best-effort and may miss a signal. It never fabricates an elevated finding.
- FIRMS, HRRR, LANDFIRE, OSM, and external API availability can affect a run. Degraded flags are part of the expected output contract.
- Historic MTBS final perimeters are not a time series. They support useful experiments but cannot alone prove arrival-time performance; timed perimeters are the relevant validation source.
- The current response water information does not invent distances for waterbody/flowline names when the source does not provide them.
- Slack and email are log-only until their credentials are configured.

## Further reading

- [Software requirements specification](SRS_fire.md)
- [Engineering decisions](DECISIONS.md)
- [V1 interface design](docs/UI_DESIGN_V1.md)
- [V2 interface design](docs/UI_DESIGN_V2.md)
- [Configuration guide for unavailable optional integrations](config/README_MISSING.md)
