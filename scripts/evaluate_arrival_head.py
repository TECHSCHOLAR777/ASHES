"""R2 + R4: full-W GBM calibration of the R1 field on R0 y_72 labels.

Uses every Mireye role A–I (200-D). Drops only the geometric E leaks
(`dist_perim_m`, `wind_ros_ellipse_dist_m`). Retrain-ablates every role and
every field under leave-one-event-out; grouped-permutation traces the rest.

    python scripts/evaluate_arrival_head.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.arrival_eval import corpus_summary, evaluate_arrival_head, load_evaluable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.arrival_head")

DEFAULT_INPUT = Path("data/training/real_conus_2015_2023_arrival_hrrr.jsonl")
DEFAULT_ARRIVAL = Path("data/training/real_conus_2015_2023_arrival.jsonl")
DEFAULT_OUTPUT = Path("data/models/arrival_head_report.json")


def _public(report: dict) -> dict:
    skip = {"grouped_permutation_importance", "field_retrain_ablation_logo", "feature_names"}
    return {k: report[k] for k in report if k not in skip}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--arrival", type=Path, default=DEFAULT_ARRIVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_evaluable(args.input)
    if len(rows) < 20:
        raise SystemExit(f"Need >=20 evaluable_72 rows with HRRR spread; found {len(rows)} in {args.input}")

    report = evaluate_arrival_head(rows)
    report["r0"] = corpus_summary(args.arrival if args.arrival.exists() else args.input)
    report["r1"] = corpus_summary(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logo = report["logo"]
    logger.info(
        "Wrote %s full_ap=%.4f raw_p72=%.4f shuffleW=%.4f kill=%s",
        args.output,
        logo["gbm_full_w_e_engine"]["pr_auc"],
        report["rank_baselines"]["engine_p72"],
        logo["gbm_shuffled_w"]["pr_auc"],
        logo["w_kill_test_passed"],
    )
    print(json.dumps(_public(report), indent=2))


if __name__ == "__main__":
    main()
