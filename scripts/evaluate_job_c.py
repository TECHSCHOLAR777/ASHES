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
    parser.add_argument(
        "--include-fields",
        type=str,
        default=None,
        help="Comma-separated W_allowed field subset. Default: full W_allowed.",
    )
    parser.add_argument(
        "--estimator",
        choices=("logistic", "gbm"),
        default="logistic",
        help="Calibrator head. gbm = HistGradientBoosting predict_proba, same Brier protocol.",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="GBM only: LOGO permute each W_allowed field on the test fold (OOS ablation).",
    )
    parser.add_argument(
        "--refit-kept",
        action="store_true",
        help="After ablation, re-run GBM LOGO on fields that earned keep.",
    )
    args = parser.parse_args()
    rows = load_evaluable_job_c(args.input)
    logger.info("evaluable rows %d events %d", len(rows), len({r["event_id"] for r in rows}))
    include = None
    if args.include_fields:
        include = [s.strip() for s in args.include_fields.split(",") if s.strip()]
    report = evaluate_job_c(
        rows,
        include_fields=include,
        estimator=args.estimator,
        ablate=args.ablate,
    )
    if args.refit_kept and report.get("w_ablation_keep"):
        logger.info("refitting GBM on ablated keep %s", report["w_ablation_keep"])
        sub = evaluate_job_c(
            rows,
            include_fields=report["w_ablation_keep"],
            estimator=args.estimator,
            ablate=False,
        )
        report["scores_ablated_subset"] = sub["scores"]
        report["deltas_ablated_subset"] = sub["deltas"]
        report["w_ablated_subset_fields"] = sub["w_allowed_fields"]
        report["w_ablated_subset_dim"] = sub["w_allowed_dim"]
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
            "w_ablation_keep",
            "engines_seen",
            "estimator",
        )
        if k in report
    }
    print(json.dumps(_json_safe(slim), indent=2))
    logger.info("wrote %s kill=%s", args.output, report["deltas"]["w_kill_test_passed"])


if __name__ == "__main__":
    main()
