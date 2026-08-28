"""Job A for Job C: LANDFIRE + HRRR-at-seed + ELMFIRE. Huygens is a hard fail.

    SPREAD_ENGINE_BIN=spread_service/elmfire_bin.py \\
    SPREAD_ENGINE_REQUIRED=1 \\
    ELMFIRE_BIN=/home/ubuntu/elmfire/build/linux/bin/elmfire \\
    SPREAD_SERVICE_PORT=8766 \\
    python scripts/enrich_job_c_elmfire.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.clients.hrrr import HRRRClient
from src.clients.landfire import LANDFIREClient
from src.clients.timed_perimeters import load_job_c_tape
from src.model.arrival_labels import seed_rings, seed_time
from src.model.job_c_sample import snapshot_for_horizon
from src.spread.historic import spread_field_for_job_c

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.job_c_elmfire")

DEFAULT_IN = Path("data/training/job_c_2023/book.jsonl")
DEFAULT_OUT = Path("data/training/job_c_2023/book_elmfire.jsonl")
ADAPTER = Path(__file__).resolve().parent.parent / "spread_service" / "elmfire_bin.py"
DEFAULT_ELMFIRE = Path("/home/ubuntu/elmfire/build/linux/bin/elmfire")


def _configure_engine() -> None:
    os.environ.setdefault("SPREAD_ENGINE_BIN", str(ADAPTER))
    os.environ["SPREAD_ENGINE_REQUIRED"] = "1"
    os.environ.setdefault("SPREAD_ENGINE_TIMEOUT_S", "1800")
    os.environ.setdefault("SPREAD_SERVICE_PORT", "8766")
    if "ELMFIRE_BIN" not in os.environ and DEFAULT_ELMFIRE.exists():
        os.environ["ELMFIRE_BIN"] = str(DEFAULT_ELMFIRE)
    if not Path(os.environ["SPREAD_ENGINE_BIN"]).exists():
        raise RuntimeError(f"SPREAD_ENGINE_BIN missing: {os.environ['SPREAD_ENGINE_BIN']}")
    if not Path(os.environ.get("ELMFIRE_BIN") or "").exists():
        raise RuntimeError("ELMFIRE_BIN is not set to a compiled elmfire binary")
    os.chmod(os.environ["SPREAD_ENGINE_BIN"], 0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-fires", type=int, default=None)
    args = parser.parse_args()
    _configure_engine()

    rows: list[dict] = []
    with args.input.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    done_events: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open() as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("spread_vector_elmfire") is not None:
                    done_events.add(rec["event_id"])

    by_event: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for rec in rows:
        eid = rec["event_id"]
        by_event[eid].append(rec)
        if eid not in order:
            order.append(eid)
    if args.max_fires is not None:
        order = order[: args.max_fires]

    landfire = LANDFIREClient()
    hrrr = HRRRClient(max_hours_back=12)
    n_write = n_field = n_hrrr_fail = n_engine_fail = 0
    fail_log = args.output.with_name("elmfire_failures.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out, fail_log.open(
        "a", encoding="utf-8"
    ) as fails:
        for i, event_id in enumerate(order, start=1):
            group = by_event[event_id]
            if args.resume and event_id in done_events:
                continue
            series = load_job_c_tape(event_id)
            field = None
            weather_src = "none"
            engine = None
            if series is None:
                logger.info("[%d/%d] %s no tape", i, len(order), event_id)
            else:
                rings = seed_rings(series)
                horizon = snapshot_for_horizon(series, 72.0)
                aoi_rings = horizon.geometry_rings if horizon is not None else rings
                t_seed = seed_time(series)
                lat0 = group[0]["site_lat"]
                lng0 = group[0]["site_lng"]
                # Prefer a vertex of the seed for HRRR, not a random sample site.
                if rings and rings[0]:
                    lng0, lat0 = rings[0][0][0], rings[0][0][1]
                weather = {
                    "wind_u": 0.0,
                    "wind_v": 0.0,
                    "rh_pct": None,
                    "temp_c": None,
                    "valid_time": t_seed.isoformat() if t_seed else None,
                }
                if t_seed is not None:
                    try:
                        wx = hrrr.get_weather(lat0, lng0, site_id=event_id, at=t_seed)
                        weather = {
                            "wind_u": wx.wind_u_10m,
                            "wind_v": wx.wind_v_10m,
                            "rh_pct": wx.rh_pct,
                            "temp_c": wx.temp_c,
                            "valid_time": wx.hrrr_valid_time,
                        }
                        weather_src = "hrrr_at_seed"
                    except Exception as exc:
                        n_hrrr_fail += 1
                        logger.warning("HRRR failed for %s: %s", event_id, exc)
                        weather_src = "hrrr_fail"
                if weather_src != "hrrr_at_seed":
                    logger.info("[%d/%d] %s skip ELMFIRE (need HRRR-at-seed, got %s)", i, len(order), event_id, weather_src)
                else:
                    field = spread_field_for_job_c(event_id, lat0, lng0, rings, aoi_rings, landfire, weather)
                    if field is None:
                        n_engine_fail += 1
                    else:
                        n_field += 1
                        engine = field.engine
                        if "huygens" in str(engine).lower():
                            raise RuntimeError(f"Huygens leaked into Job C for {event_id}")
            if field is None:
                fails.write(
                    json.dumps(
                        {
                            "event_id": event_id,
                            "weather_source": weather_src,
                            "engine": engine,
                            "n_sites": len(group),
                        }
                    )
                    + "\n"
                )
                fails.flush()
                logger.info(
                    "[%d/%d] %s weather=%s field=False (not written; resume will retry)",
                    i, len(order), event_id, weather_src,
                )
                continue
            for rec in group:
                out_rec = dict(rec)
                out_rec["weather_source"] = weather_src
                out_rec["engine"] = engine
                sample = field.sample(rec["site_lat"], rec["site_lng"])
                out_rec["spread_vector_elmfire"] = [
                    sample.eta_hours,
                    sample.eta_sigma_hours,
                    sample.p_burn_24,
                    sample.p_burn_48,
                    sample.p_burn_72,
                ]
                out_rec["spread_field_version"] = field.spread_field_version
                out_rec["inside_aoi"] = sample.inside_aoi
                out.write(json.dumps(out_rec) + "\n")
                n_write += 1
            logger.info(
                "[%d/%d] %s weather=%s field=%s engine=%s rows=%d",
                i, len(order), event_id, weather_src, True, engine, len(group),
            )
            out.flush()
    logger.info(
        "Wrote %d rows field_fires=%d hrrr_fail=%d engine_fail=%d -> %s",
        n_write, n_field, n_hrrr_fail, n_engine_fail, args.output,
    )


if __name__ == "__main__":
    main()
