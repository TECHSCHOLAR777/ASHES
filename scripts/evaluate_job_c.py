"""Evaluate Job C: logistic P(y|engine) vs P(y|engine, W_allowed). Event-held-out.

    python scripts/evaluate_job_c.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.job_c_eval import evaluate_job_c, load_evaluable_job_c

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.job_c_eval")

DEFAULT_IN = Path("data/training/job_c_2023/book_elmfire_w.jsonl")
DEFAULT_OUT = Path("data/models/job_c_report.json")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = load_evaluable_job_c(args.input)
    logger.info("evaluable rows %d events %d", len(rows), len({r["event_id"] for r in rows}))
    report = evaluate_job_c(rows)
    engines = {r.get("engine") for r in rows}
    report["engines_seen"] = sorted(str(e) for e in engines)
    report["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    slim = {
        k: report[k]
        for k in (
            "n_evaluable",
            "n_events",
            "n_pos",
            "n_neg",
            "pos_rate",
            "w_allowed_dim",
            "scores",
            "deltas",
            "engines_seen",
        )
        if k in report
    }
    print(json.dumps(_json_safe(slim), indent=2))
    logger.info("wrote %s kill=%s", args.output, report["deltas"]["w_kill_test_passed"])


if __name__ == "__main__":
    main()
