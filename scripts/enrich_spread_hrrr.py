"""R1: re-run historic spread from the first operational/IR seed + HRRR-at-seed.

    python scripts/enrich_spread_hrrr.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.clients.hrrr import HRRRClient
from src.clients.landfire import LANDFIREClient
from src.clients.mtbs import MTBSClient
from src.clients.timed_perimeters import load_series_cache
from src.model.arrival_labels import seed_rings, seed_time
from src.spread.historic import HISTORIC_DEFAULT_WIND_U, HISTORIC_DEFAULT_WIND_V, spread_field_for_timed_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.enrich_hrrr")

DEFAULT_INPUT = Path("data/training/real_conus_2015_2023_arrival.jsonl")
DEFAULT_OUTPUT = Path("data/training/real_conus_2015_2023_arrival_hrrr.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-fires", type=int, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    with args.input.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open() as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("spread_vector_hrrr"):
                    done.add(rec["site_id"])

    by_event: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for rec in rows:
        eid = rec["event_id"]
        by_event[eid].append(rec)
        if eid not in order:
            order.append(eid)
    if args.max_fires is not None:
        order = order[: args.max_fires]

    mtbs = MTBSClient()
    landfire = LANDFIREClient()
    hrrr = HRRRClient(max_hours_back=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_write = n_field = n_hrrr_fail = 0

    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for i, event_id in enumerate(order, start=1):
            group = by_event[event_id]
            if args.resume and all(rec["site_id"] in done for rec in group):
                continue
            series = load_series_cache(event_id)
            fires = mtbs.get_fires_by_event_ids([event_id])
            fire = fires[0] if fires else None
            field = None
            weather_src = "none"
            if fire is None:
                logger.warning("[%d/%d] %s no MTBS fire", i, len(order), event_id)
            else:
                rings = seed_rings(series) if series else []
                t_seed = seed_time(series)
                weather = {
                    "wind_u": HISTORIC_DEFAULT_WIND_U,
                    "wind_v": HISTORIC_DEFAULT_WIND_V,
                    "rh_pct": None,
                    "temp_c": None,
                    "valid_time": t_seed.isoformat() if t_seed else None,
                }
                weather_src = "reference_2.2ms"
                if t_seed is not None and fire.centroid_lat is not None and fire.centroid_lng is not None:
                    try:
                        wx = hrrr.get_weather(fire.centroid_lat, fire.centroid_lng, site_id=event_id, at=t_seed)
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
                        logger.warning("HRRR failed for %s at %s: %s; using reference wind", event_id, t_seed, exc)
                if series is not None and series.n_times >= 2:
                    field = spread_field_for_timed_seed(fire, landfire, rings, weather)
                    if field is not None:
                        n_field += 1
                logger.info(
                    "[%d/%d] %s weather=%s seed_rings=%d field=%s",
                    i, len(order), event_id, weather_src, len(rings), field is not None,
                )
            for rec in group:
                if rec["site_id"] in done:
                    continue
                out_rec = dict(rec)
                out_rec["weather_source"] = weather_src
                if field is not None and rec.get("site_lat") is not None:
                    sample = field.sample(float(rec["site_lat"]), float(rec["site_lng"]))
                    out_rec["spread_vector_hrrr"] = [
                        sample.eta_hours if sample.eta_hours is not None else 72.0,
                        sample.eta_sigma_hours if sample.eta_sigma_hours is not None else 24.0,
                        sample.p_burn_24 if sample.p_burn_24 is not None else 0.0,
                        sample.p_burn_48 if sample.p_burn_48 is not None else 0.0,
                        sample.p_burn_72 if sample.p_burn_72 is not None else 0.0,
                    ]
                    out_rec["spread_field_version_hrrr"] = sample.spread_field_version
                out.write(json.dumps(out_rec) + "\n")
                n_write += 1
            out.flush()
    logger.info("Wrote %d rows, fields for %d fires, hrrr_fail=%d -> %s", n_write, n_field, n_hrrr_fail, args.output)


if __name__ == "__main__":
    main()
