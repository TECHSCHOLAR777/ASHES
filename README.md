# ASHES

**(By Team OpenAEye)**

ASHES is a wildfire site-event copilot for organisations across the United States. It watches a book of known sites (campuses, warehouses, plants, yards) and turns live fire conditions plus what actually sits around each location into one cited **ActionCard**: how serious this is, and what to do next.

---

## Summary of the Idea

Wildfires become difficult for organisations when a fire approaches a facility. Alerts, satellite hotspots, weather, perimeters, terrain, roads, and site facts live in different systems. The question that matters is simple and time-sensitive: **how serious is the threat, and what should we do next?**

ASHES answers that question for a configured site, not for the public internet. It joins two kinds of evidence:

- **What is happening now:** warnings, thermal detections, incidents, perimeters, and wind.
- **What exists around the site:** fuel, terrain, hazard, access, water, and governance.

**Mireye** supplies the site-world layer: sourced geospatial facts (roads, land, exposure, infrastructure, habitat, and related fields) so the system knows not only where fire is, but what sits next to the pin.

**ELMFIRE** is the live spread engine. It is a production-grade, industry-standard wildfire model: vegetation, slope, and fire physics are native to the solve. For a live or simulated ignition it returns arrival time at the community pin and burn probability at 24, 48, and 72 hours.

A **Pareto checks policy table** (perimeter distance, red-flag weather, thermal hotspots, engine ETA and ETA uncertainty, housing density, road access, containment, and prior site state) then places the site in a bucket (`no_action`, `monitor`, `prepare`, `protect_asset`, `evacuate_site`, `inspect_after`). An **agent** uses that bucket, the engine clock, and the Mireye facts to write a cited playbook: the ActionCard. Serious buckets also get a **ResponseCard**: access, water, hazmat priorities, and responsible agency.

The result is an auditable operating picture: live signals, live site context, live physics, a deterministic bucket, and a playbook a team can act on before the threat becomes a crisis.

---

## Impact

When an organisation runs many sites, the work is triage: watch changing conditions, interpret local risk, and decide where attention goes first. ASHES makes that a repeatable workflow instead of a one-off map session.

The same evidence and the same bucket rules run across the site book, so operators can see which locations need watching, preparation, protection, or escalation. ActionCards are short enough for a handoff or a management update: action, why, engine ETA, P(burn), and sources. When the bucket is serious, ResponseCards add the operational layer: how you get there, where water is, what is hazardous, who owns the land.

ASHES does not replace incident command or an official evacuation order. It reduces the scramble of gathering fragments, and it leaves a record of what the system knew when it spoke.

---

## Technicality

ASHES is a live pipeline. Clock and probability come from ELMFIRE. The bucket comes from a Pareto checks policy table. The agent is the bridge that fetches, runs the engine, and writes the card.

**1. Live event signals.**  
NWS CAP alerts, SPC fire-weather outlook, NASA FIRMS hotspots, WFIGS incidents and operational perimeters, and HRRR wind/humidity are fetched together for the site. Failures degrade the card with flags; they do not invent a missing feed.

**2. Live Mireye fetch.**  
The Mireye Fire Intelligence Set fills site-world context: fuel and vegetation, terrain, fire-hazard class, exposure, roads and egress, compounding hazards, water, and governance. Fields are typed, cited, and cached with role-specific freshness. Mireye is a fact source, never asked as a free-text question.

**3. Live ELMFIRE.**  
ELMFIRE is a production-grade, industry-standard wildfire spread engine. For a suitable incident or a stated ignition, ASHES runs it live. Vegetation, slope, and fire physics are native to the solve. The engine returns arrival time, uncertainty, and P(burn) by 24, 48, and 72 hours at the community pin, plus an hour-by-hour field for map playback. The UI clock is that ELMFIRE sample.

**4. Pareto checks policy table.**  
A deterministic Pareto checks policy table is the only authority for the action (perimeter distance, red-flag weather, thermal hotspots, engine ETA and ETA uncertainty, housing density, road access, containment, and prior site state). The language model cannot pick or override the bucket. High uncertainty can clamp an overconfident evacuate down to protect.

**5. Agentic ActionCard.**  
The agent orchestrates the live tools: parse the place, pull event feeds and Mireye, run ELMFIRE, apply the Pareto checks, then write the playbook from the bucket plus cited Mireye fields (agency, roads, water, only if fetched). A validator rejects new numbers or a severity the policy did not choose. On protect or evacuate, a separate response path adds water, access, hazmat, and agency into a ResponseCard.

Operationally: SQLite for site state and role cache, daily JSONL audit logs, log-only Slack/email until credentials exist, and cards that carry source URLs, vintages, policy version, spread-field version, and degraded flags.

```text
  site pin + question
           |
           v
  live E (NWS, FIRMS, WFIGS, HRRR)     live W (Mireye)
           |                                    |
           +----------------+-------------------+
                            v
                   live ELMFIRE
              ETA · P(burn 24/48/72) · hour field
                            |
                            v
         Pareto checks policy table  ->  bucket
                            |
                            v
         agent writes cited ActionCard / playbook
              (+ ResponseCard when escalated)
                            |
                            v
              UI · Slack/email · JSONL audit
```

---

## Getting started

Python 3.11+ (developed on 3.12). You need Mireye key(s), optional FIRMS `MAP_KEY`, OpenAI for the agentic loop, and a compiled ELMFIRE binary for the live engine (`ELMFIRE_BIN`, `SPREAD_ENGINE_BIN`, `SPREAD_ENGINE_REQUIRED=1`).

```bash
git clone <repository-url>
cd ASHES
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Fill `.env`. Keep it out of git.

```bash
python scripts/ask.py --agentic --simulate --lat 33.7461 --lng -116.7139 \
  --q "Idyllwild is the community. If chaparral west of town ignites, is the town in play?"
python scripts/serve.py
```

UI: `http://127.0.0.1:8080`. Watch, Ask, Simulator, Audit.

```bash
pytest
```

- `data/logs/YYYY-MM-DD.jsonl`: tool calls, policy, delivery (no raw keys).
- `data/cache/w_cache.db` / `site_state.db`: Mireye cache and watch state.

## Repository

| Path | Purpose |
| --- | --- |
| `src/agents/` | Agentic loop, tools, playbooks, response dossier |
| `src/clients/` | NWS, FIRMS, WFIGS, HRRR, Mireye, USGS, OSM |
| `src/policy/` | Pareto checks policy table |
| `src/spread/` + `spread_service/` | HTTP `spread_run`; ELMFIRE adapter |
| `src/serve/` | FastAPI UI |
| `src/schemas/` + `src/validator/` | ActionCard / ResponseCard; grounded brief |
| `config/` | Sites, policy, showcase case, field catalogue |
| `tests/` | Policy, clients, spread, agent boundaries |
