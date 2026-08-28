"""Sample Job C sites from downloaded Daily tapes and attach R0 arrival labels.

    python scripts/build_job_c_book.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clients.timed_perimeters import JOB_C_TAPES, load_job_c_tape
from src.model.job_c_sample import early_tape_ok, labeled_rows, sample_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.job_c_book")

INVENTORY = Path("data/training/job_c_2023/tape_inventory.json")
OUT = Path("data/training/job_c_2023/book.jsonl")
SUMMARY = Path("data/training/job_c_2023/book_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-fires", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    fire_ids = list(inv["usable_fire_ids"])
    if args.max_fires is not None:
        fire_ids = fire_ids[: args.max_fires]

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open() as handle:
            for line in handle:
                rec = json.loads(line)
                done.add(rec["event_id"])

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_write = n_eval = n_pos = n_skip_tape = n_skip_early = 0
    used_events = 0
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for i, fire_id in enumerate(fire_ids, start=1):
            if fire_id in done:
                continue
            series = load_job_c_tape(fire_id)
            if series is None or series.n_times < 4:
                n_skip_tape += 1
                logger.info("[%d/%d] %s skip (no tape on disk)", i, len(fire_ids), fire_id)
                continue
            if not early_tape_ok(series, 96.0):
                n_skip_early += 1
                logger.info("[%d/%d] %s skip (no second ring in first 96h)", i, len(fire_ids), fire_id)
                continue
            sites = sample_series(series, rng)
            rows = labeled_rows(series, sites)
            n_eval_fire = n_pos_fire = 0
            for rec in rows:
                out.write(json.dumps(rec) + "\n")
                n_write += 1
                if rec["arrival"].get("evaluable_72"):
                    n_eval += 1
                    n_eval_fire += 1
                    if rec["arrival"].get("y_72") == 1:
                        n_pos += 1
                        n_pos_fire += 1
            used_events += 1
            logger.info(
                "[%d/%d] %s sites=%d eval72=%d pos=%d n_times=%d",
                i, len(fire_ids), fire_id, len(rows), n_eval_fire, n_pos_fire, series.n_times,
            )
            out.flush()
    SUMMARY.write_text(
        json.dumps(
            {
                "n_rows": n_write,
                "n_events_written": used_events,
                "evaluable_72": n_eval,
                "y_72_pos": n_pos,
                "skipped_no_tape": n_skip_tape,
                "skipped_sparse_early_tape": n_skip_early,
                "output": str(args.output),
                "tape_dir": str(JOB_C_TAPES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d rows / %d events (eval72=%d pos=%d skip_tape=%d skip_early=%d) -> %s",
        n_write, used_events, n_eval, n_pos, n_skip_tape, n_skip_early, args.output,
    )


if __name__ == "__main__":
    main()
