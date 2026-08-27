"""Path B: attach delegated spread_vector to an existing V1 jsonl (no Mireye credits).

Reads `data/training/real_conus_2015_2023.jsonl` (real W/E already paid for) and writes
a sidecar with `spread_vector` from LANDFIRE + out-of-process `spread_run` once per
fire. Does not call Mireye. `--resume` skips site_ids that already have a spread_vector
in the output file.

    python scripts/enrich_spread_vectors.py
    python scripts/enrich_spread_vectors.py --max-fires 20 --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.clients.landfire import LANDFIREClient  # noqa: E402
from src.clients.mtbs import MTBSClient  # noqa: E402
from src.features.e_packer import E_VECTOR_FIELD_ORDER  # noqa: E402
from src.model.training_data import (  # noqa: E402
    SampleBuildResult,
    attach_spread_vector,
    reconstruct_sample_point,
)
from src.spread.historic import spread_field_for_fire  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.enrich_spread_vectors")

DIST_IDX = E_VECTOR_FIELD_ORDER.index("dist_perim_m")
ROS_IDX = E_VECTOR_FIELD_ORDER.index("wind_ros_ellipse_dist_m")

DEFAULT_INPUT = Path("data/training/real_conus_2015_2023.jsonl")
DEFAULT_OUTPUT = Path("data/training/real_conus_2015_2023_with_spread.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach LANDFIRE+spread_run spread_vector to an existing V1 jsonl "
        "without re-fetching Mireye (Path B / AC-12)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-fires",
        type=int,
        default=None,
        help="Cap unique event_ids (first-seen order). Default: all fires in the input.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip site_ids that already have a spread_vector in --output.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _done_site_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            spread = rec.get("spread_vector")
            if rec.get("site_id") and isinstance(spread, list) and len(spread) >= 5:
                done.add(rec["site_id"])
    return done


def _row_to_result(rec: dict) -> SampleBuildResult:
    return SampleBuildResult(
        site_id=rec["site_id"],
        event_id=rec["event_id"],
        t0=rec["t0"],
        w_vector=rec["w_vector"],
        w_mask=rec["w_mask"],
        e_vector=rec["e_vector"],
        y=int(rec["y"]),
        dist_source=rec.get("dist_source") or "ignition_centroid_proxy",
        spread_vector=rec.get("spread_vector"),
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input jsonl not found: {args.input}")

    rows = _load_jsonl(args.input)
    by_event: dict[str, list[dict]] = defaultdict(list)
    event_order: list[str] = []
    for rec in rows:
        eid = rec["event_id"]
        if eid not in by_event:
            event_order.append(eid)
        by_event[eid].append(rec)

    if args.max_fires is not None:
        event_order = event_order[: max(args.max_fires, 0)]

    done = _done_site_ids(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if (args.resume and args.output.exists()) else "w"

    mtbs = MTBSClient()
    try:
        fires = {f.event_id: f for f in mtbs.get_fires_by_event_ids(event_order)}
    finally:
        mtbs.close()
    logger.info(
        "MTBS returned %d/%d requested fires; %d samples already have spread_vector",
        len(fires),
        len(event_order),
        len(done),
    )

    landfire = LANDFIREClient()
    n_written = 0
    n_labeled = 0
    n_failed_fires = 0
    try:
        with open(args.output, write_mode, encoding="utf-8", buffering=1) as out:
            for i, event_id in enumerate(event_order, start=1):
                samples = by_event[event_id]
                pending = [rec for rec in samples if rec["site_id"] not in done]
                if not pending:
                    continue
                fire = fires.get(event_id)
                field = spread_field_for_fire(fire, landfire) if fire is not None else None
                if field is None:
                    n_failed_fires += 1
                    logger.warning(
                        "No spread field for %s (%s); writing unlabeled rows",
                        event_id,
                        "mtbs miss" if fire is None else "LANDFIRE/spread_run failed",
                    )
                for rec in pending:
                    result = _row_to_result(rec)
                    coords_source = None
                    site_lat = rec.get("site_lat")
                    site_lng = rec.get("site_lng")
                    if field is not None and fire is not None:
                        e_vec = rec.get("e_vector") or []
                        dist_m = float(e_vec[DIST_IDX]) if len(e_vec) > DIST_IDX else 0.0
                        ros_target = float(e_vec[ROS_IDX]) if len(e_vec) > ROS_IDX else None
                        point = reconstruct_sample_point(
                            fire, dist_m, int(rec["y"]), ros_target=ros_target
                        )
                        if point is not None:
                            site_lat, site_lng = point
                            coords_source = "reconstructed_centroid_circle"
                            result = attach_spread_vector(result, field.sample(site_lat, site_lng))
                    payload = {
                        "site_id": result.site_id,
                        "event_id": result.event_id,
                        "t0": result.t0,
                        "w_vector": result.w_vector,
                        "w_mask": result.w_mask,
                        "e_vector": result.e_vector,
                        "y": result.y,
                        "dist_source": result.dist_source,
                        "spread_vector": result.spread_vector,
                    }
                    if site_lat is not None and site_lng is not None:
                        payload["site_lat"] = site_lat
                        payload["site_lng"] = site_lng
                    if coords_source:
                        payload["coords_source"] = coords_source
                    out.write(json.dumps(payload) + "\n")
                    n_written += 1
                    if result.spread_vector is not None:
                        n_labeled += 1
                        done.add(result.site_id)
                logger.info(
                    "Fire %d/%d %s: labeled_this_run=%d field=%s",
                    i,
                    len(event_order),
                    event_id,
                    sum(1 for rec in pending if rec["site_id"] in done),
                    "ok" if field is not None else "none",
                )
    finally:
        landfire.close()

    logger.info(
        "Done: wrote %d rows (%d with spread_vector, %d fires without a field) to %s",
        n_written,
        n_labeled,
        n_failed_fires,
        args.output,
    )


if __name__ == "__main__":
    main()
