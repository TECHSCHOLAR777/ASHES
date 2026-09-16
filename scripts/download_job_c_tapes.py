"""Download WFIGS Daily geometries for Job C usable UniqueFireIdentifier series.

    python scripts/download_job_c_tapes.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clients.timed_perimeters import (
    TimedPerimeterClient,
    job_c_tape_path,
    load_job_c_tape,
    save_job_c_tape,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.job_c_tapes")

INVENTORY = Path("data/training/job_c_2023/tape_inventory.json")
SUMMARY = Path("data/training/job_c_2023/tape_download_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-fires", type=int, default=None)
    args = parser.parse_args()

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    fire_ids = list(inv["usable_fire_ids"])
    if args.max_fires is not None:
        fire_ids = fire_ids[: args.max_fires]

    timed = TimedPerimeterClient(timeout=120.0)
    n_ok = n_empty = n_skip = 0
    rows = []
    try:
        for i, fire_id in enumerate(fire_ids, start=1):
            path = job_c_tape_path(fire_id)
            if args.resume and path.exists():
                series = load_job_c_tape(fire_id)
                n_skip += 1
            else:
                series = timed.fetch_series_by_fire_id(fire_id, site_id=fire_id, daily_only=True)
                save_job_c_tape(fire_id, series)
            if series is None or series.n_times < 2:
                n_empty += 1
                logger.info("[%d/%d] %s empty", i, len(fire_ids), fire_id)
                continue
            n_ok += 1
            snaps = sorted(series.snapshots, key=lambda s: s.t)
            span = (snaps[-1].t - snaps[0].t).total_seconds() / 3600.0
            methods = sorted({s.map_method for s in snaps if s.map_method})
            rows.append(
                {
                    "fire_id": fire_id,
                    "name": series.name,
                    "n_times": series.n_times,
                    "n_snapshots": len(snaps),
                    "span_hours": span,
                    "max_acres": max((s.acres or 0.0) for s in snaps),
                    "map_methods": methods,
                    "seed_time": snaps[0].t.isoformat(),
                    "last_time": snaps[-1].t.isoformat(),
                }
            )
            logger.info(
                "[%d/%d] %s n_times=%d span=%.0fh acres=%.0f methods=%s",
                i, len(fire_ids), fire_id, series.n_times, span,
                max((s.acres or 0.0) for s in snaps), methods[:3],
            )
    finally:
        timed.close()
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        json.dumps(
            {
                "n_requested": len(fire_ids),
                "n_ok": n_ok,
                "n_empty": n_empty,
                "n_cache_hit": n_skip,
                "fires": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("tapes ok=%d empty=%d cache=%d summary=%s", n_ok, n_empty, n_skip, SUMMARY)


if __name__ == "__main__":
    main()
