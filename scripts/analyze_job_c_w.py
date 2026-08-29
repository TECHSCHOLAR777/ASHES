"""Rank W_allowed by event-level residual correlation, then Job C LOGO on the subset.

    python scripts/analyze_job_c_w.py
    python scripts/analyze_job_c_w.py --eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.job_c_eval import evaluate_job_c, load_evaluable_job_c
from src.model.job_c_w_select import correlation_map, gbm_field_ranks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.job_c_w_select")

DEFAULT_IN = Path("data/training/job_c_2023/book_elmfire_w.jsonl")
DEFAULT_CORR = Path("data/models/job_c_w_correlation.json")
DEFAULT_REPORT = Path("data/models/job_c_report.json")


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


def _heatmap(fields: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        logger.warning("skip heatmap: %s", exc)
        return
    names = [f["field"] for f in fields]
    mat = np.array(
        [
            [f["rho_y"], f["rho_eta"], f["rho_between_event"], f["rho_residual"]]
            for f in fields
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.5, max(6.0, 0.22 * len(names))))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["ρ vs y", "ρ vs η", "ρ between-fire", "ρ vs residual (within)"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Job C W_allowed event-level Spearman")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--correlation-out", type=Path, default=DEFAULT_CORR)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-gbm-rank", action="store_true")
    args = parser.parse_args()

    rows = load_evaluable_job_c(args.input)
    logger.info("evaluable rows %d events %d", len(rows), len({r["event_id"] for r in rows}))
    cmap = correlation_map(rows)
    if not args.skip_gbm_rank:
        cmap["gbm_residual_ranks"] = gbm_field_ranks(rows)
        cmap["gbm_note"] = (
            "HistGradientBoosting on LOGO engine residual is a ranking diagnostic only; "
            "the Job C head remains logistic. not_a_218d_gbm still true."
        )
    args.correlation_out.parent.mkdir(parents=True, exist_ok=True)
    args.correlation_out.write_text(json.dumps(_json_safe(cmap), indent=2), encoding="utf-8")
    logger.info(
        "FDR fields %s; head_fields %s -> %s",
        cmap["selected_fdr_fields"],
        cmap["head_fields"],
        args.correlation_out,
    )
    _heatmap(cmap["fields"], args.correlation_out.with_suffix(".png"))

    if not args.eval:
        print(json.dumps(_json_safe({
            "n_events": cmap["n_events"],
            "selected_fdr_fields": cmap["selected_fdr_fields"],
            "head_fields": cmap["head_fields"],
            "head_rule": cmap["head_rule"],
            "top_fields": cmap["fields"][:12],
        }), indent=2))
        return

    prior = {}
    if args.eval_output.exists():
        prior = json.loads(args.eval_output.read_text(encoding="utf-8"))
    report = evaluate_job_c(rows, include_fields=cmap["head_fields"])
    report["engines_seen"] = sorted({str(r.get("engine")) for r in rows})
    report["input"] = str(args.input)
    report["w_selection"] = {
        "correlation": str(args.correlation_out),
        "rule": cmap["head_rule"],
        "selected_fdr_fields": cmap["selected_fdr_fields"],
        "head_fields": cmap["head_fields"],
        "not_a_218d_gbm": True,
    }
    if prior.get("scores") and prior.get("w_include_fields") is None and prior.get("w_allowed_dim", 0) > 20:
        report["scores_full_w_allowed"] = prior["scores"]
        report["deltas_full_w_allowed"] = prior.get("deltas")
        report["n_events_full_w_allowed"] = prior.get("n_events")
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    slim = {
        k: report[k]
        for k in (
            "n_evaluable",
            "n_events",
            "w_allowed_dim",
            "w_allowed_fields",
            "scores",
            "deltas",
            "w_selection",
        )
        if k in report
    }
    print(json.dumps(_json_safe(slim), indent=2))
    logger.info("wrote %s kill=%s", args.eval_output, report["deltas"]["w_kill_test_passed"])


if __name__ == "__main__":
    main()
