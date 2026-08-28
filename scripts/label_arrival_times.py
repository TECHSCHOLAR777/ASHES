"""R0: attach timed-perimeter arrival labels to the Path B sidecar.

    python scripts/label_arrival_times.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clients.mtbs import MTBSClient
from src.clients.timed_perimeters import (
    TimedPerimeterClient,
    load_series_cache,
    save_series_cache,
    series_cache_exists,
)
from src.model.arrival_labels import ArrivalLabel, label_point
from src.model.training_data import ignition_datetime
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.label_arrival")

DEFAULT_INPUT = Path("data/training/real_conus_2015_2023_with_spread.jsonl")
DEFAULT_OUTPUT = Path("data/training/real_conus_2015_2023_arrival.jsonl")


def _label_to_dict(lab: ArrivalLabel) -> dict:
    return {
        "n_snapshots": lab.n_snapshots,
        "n_times": lab.n_times,
        "source": lab.source,
        "timed_fire_id": lab.fire_id,
        "seed_time": lab.seed_time,
        "last_time": lab.last_time,
        "already_burned_at_seed": lab.already_burned_at_seed,
        "arrival_hours": lab.arrival_hours,
        "hit_time": lab.hit_time,
        "series_hours": lab.series_hours,
        "y_24": lab.y_24,
        "y_48": lab.y_48,
        "y_72": lab.y_72,
        "evaluable_72": lab.evaluable_72,
        "map_methods": lab.map_methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-fires", type=int, default=None)
    parser.add_argument("--radius-km", type=float, default=25.0)
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
                done.add(rec["site_id"])

    event_order: list[str] = []
    by_event: dict[str, list[dict]] = {}
    for rec in rows:
        eid = rec["event_id"]
        by_event.setdefault(eid, []).append(rec)
        if eid not in by_event or eid not in event_order:
            if eid not in event_order:
                event_order.append(eid)
    if args.max_fires is not None:
        event_order = event_order[: args.max_fires]

    mtbs = MTBSClient()
    timed = TimedPerimeterClient()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_eval = n_write = n_no_series = 0

    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for i, event_id in enumerate(event_order, start=1):
            group = by_event[event_id]
            if args.resume and all(rec["site_id"] in done for rec in group):
                continue
            fires = mtbs.get_fires_by_event_ids([event_id])
            fire = fires[0] if fires else None
            t0 = ignition_datetime(fire) if fire else None
            year = None
            if fire and fire.ignition_date:
                year = fire.ignition_date.year
            elif group[0].get("t0"):
                year = datetime.fromisoformat(group[0]["t0"]).year
            if series_cache_exists(event_id):
                series = load_series_cache(event_id)
            else:
                series = None
                if fire and fire.centroid_lat is not None and fire.centroid_lng is not None and year:
                    series = timed.fetch_series(
                        fire.centroid_lat,
                        fire.centroid_lng,
                        year,
                        site_id=event_id,
                        radius_km=args.radius_km,
                    )
                save_series_cache(event_id, series)
            if series is None:
                n_no_series += 1
                logger.info("[%d/%d] %s no timed series", i, len(event_order), event_id)
            else:
                logger.info(
                    "[%d/%d] %s series=%s n_times=%d methods=%s",
                    i,
                    len(event_order),
                    event_id,
                    series.source,
                    series.n_times,
                    sorted({s.map_method for s in series.snapshots if s.map_method}),
                )
            for rec in group:
                if rec["site_id"] in done:
                    continue
                lat, lng = rec.get("site_lat"), rec.get("site_lng")
                lab = label_point(float(lat), float(lng), series) if lat is not None and lng is not None else ArrivalLabel(
                    n_snapshots=0, n_times=0, source="no_coords", fire_id=None, seed_time=None,
                    last_time=None, already_burned_at_seed=False, arrival_hours=None, hit_time=None,
                    series_hours=None, y_24=None, y_48=None, y_72=None, evaluable_72=False, map_methods=[],
                )
                out_rec = dict(rec)
                out_rec["arrival"] = _label_to_dict(lab)
                if t0:
                    out_rec["ignition_t0"] = t0.isoformat()
                out.write(json.dumps(out_rec) + "\n")
                n_write += 1
                if lab.evaluable_72:
                    n_eval += 1
            out.flush()
    timed.close()
    logger.info("Wrote %d rows (%d evaluable_72, %d fires without a series) to %s", n_write, n_eval, n_no_series, args.output)


if __name__ == "__main__":
    main()
