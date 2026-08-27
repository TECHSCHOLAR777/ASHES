"""Builds a real training set from MTBS fires (SRS 6.2) - not synthetic, not hardcoded.

    python scripts/build_training_set.py --bbox -124 32 -114 42 \\
        --year-start 2018 --year-end 2023 --min-acres 5000 \\
        --positives-per-fire 3 --hard-negatives-per-fire 3 --easy-negatives-per-fire 2 \\
        --output data/training/real_2018_2023_ca.jsonl

Read `src/model/training_data.py`'s module docstring before trusting the output for a real
H1 claim - it documents exactly which fields are honestly reconstructed vs. proxied (e.g.
`dist_perim_m` is distance to the fire's ignition centroid, not a true t0 perimeter, because
MTBS does not publish a perimeter time series).

This makes real Mireye `fetch` calls (one per sample) plus free FIRMS-archive and
HRRR-archive calls. Runs slowly by design (Mireye's 300 rpm/key limiter still applies) and
is meant to be run deliberately, not as part of the watch loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.clients.firms import FIRMSClient  # noqa: E402
from src.clients.hrrr import HRRRClient  # noqa: E402
from src.clients.mireye import MireyeClient  # noqa: E402
from src.clients.mtbs import MTBSClient  # noqa: E402
from src.model.training_data import (  # noqa: E402
    build_sample,
    sample_easy_negative_points,
    sample_hard_negative_points,
    sample_positive_points,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.build_training_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real MTBS-labeled training set")
    parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("MIN_LNG", "MIN_LAT", "MAX_LNG", "MAX_LAT"))
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--min-acres", type=float, default=1000.0, help="MTBS's own West-region size floor (SRS 6.1)")
    parser.add_argument("--max-fires", type=int, default=50)
    parser.add_argument("--positives-per-fire", type=int, default=3)
    parser.add_argument("--hard-negatives-per-fire", type=int, default=3)
    parser.add_argument("--easy-negatives-per-fire", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    keys = [os.environ.get(f"MIREYE_KEY_{i}") for i in (1, 2, 3)]
    keys = [k for k in keys if k]
    if not keys:
        raise RuntimeError("No MIREYE_KEY_* found in environment")

    mireye = MireyeClient(keys)
    firms = FIRMSClient()
    hrrr = HRRRClient()
    mtbs = MTBSClient()

    min_lng, min_lat, max_lng, max_lat = args.bbox
    fires = mtbs.get_fires_in_bbox(
        min_lng, min_lat, max_lng, max_lat,
        min_acres=args.min_acres, year_start=args.year_start, year_end=args.year_end,
        max_features=args.max_fires,
    )
    logger.info("Found %d MTBS fires matching the filter", len(fires))

    all_centroids = [
        (f.centroid_lat, f.centroid_lng) for f in fires if f.centroid_lat is not None and f.centroid_lng is not None
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for fire in fires:
            positives = sample_positive_points(fire, args.positives_per_fire, rng)
            hard_negatives = sample_hard_negative_points(fire, args.hard_negatives_per_fire, rng)
            other_centroids = [c for c in all_centroids if c != (fire.centroid_lat, fire.centroid_lng)]
            easy_negatives = sample_easy_negative_points(other_centroids, args.easy_negatives_per_fire, rng)

            labeled_points = (
                [(lat, lng, 1) for lat, lng in positives]
                + [(lat, lng, 0) for lat, lng in hard_negatives]
                + [(lat, lng, 0) for lat, lng in easy_negatives]
            )

            for i, (lat, lng, label) in enumerate(labeled_points):
                sample_id = f"{fire.event_id}_{i:03d}"
                sample = build_sample(mireye, firms, hrrr, fire, lat, lng, label, sample_id)
                if sample is None:
                    continue
                out.write(
                    json.dumps(
                        {
                            "site_id": sample.site_id,
                            "event_id": sample.event_id,
                            "t0": sample.t0,
                            "w_vector": sample.w_vector,
                            "w_mask": sample.w_mask,
                            "e_vector": sample.e_vector,
                            "y": sample.y,
                        }
                    )
                    + "\n"
                )
                n_written += 1

            logger.info("Fire %s (%s): wrote %d samples so far", fire.event_id, fire.incident_name, n_written)

    logger.info("Done: wrote %d real training samples to %s", n_written, output_path)


if __name__ == "__main__":
    main()
