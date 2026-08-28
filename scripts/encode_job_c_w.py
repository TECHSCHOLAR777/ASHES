"""Fetch Mireye W for Job C sites (new coordinates, new spend). Head uses W_allowed.

    python scripts/encode_job_c_w.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.clients.mireye import MireyeClient
from src.features.w_encoder import encode_w, load_field_catalog, ordered_model_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.job_c_w")

DEFAULT_IN = Path("data/training/job_c_2023/book_elmfire.jsonl")
DEFAULT_OUT = Path("data/training/job_c_2023/book_elmfire_w.jsonl")


def _keys() -> list[str]:
    keys = [os.environ.get(f"MIREYE_KEY_{i}") for i in (1, 2, 3)]
    keys = [k for k in keys if k]
    if not keys:
        raise RuntimeError("No MIREYE_KEY_* in environment")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    catalog = load_field_catalog()
    fields = [name for _role, name, _meta in ordered_model_fields(catalog)]
    client = MireyeClient(_keys())

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open() as handle:
            for line in handle:
                rec = json.loads(line)
                if rec.get("w_vector"):
                    done.add(rec["site_id"])

    n_in = n_write = n_skip = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as handle, args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_in += 1
            if args.max_rows is not None and n_write >= args.max_rows:
                break
            if rec["site_id"] in done:
                n_skip += 1
                continue
            raw = client.fetch(rec["site_lat"], rec["site_lng"], fields, site_id=rec["site_id"])
            w = encode_w(raw, catalog)
            rec["w_vector"] = w.vector.tolist()
            rec["w_mask"] = w.mask.tolist()
            rec["w_feature_names"] = w.feature_names
            rec["w_vintages"] = w.vintages
            out.write(json.dumps(rec) + "\n")
            n_write += 1
            if n_write % 25 == 0:
                logger.info("encoded %d rows", n_write)
                out.flush()
    logger.info("read=%d wrote=%d skipped=%d -> %s", n_in, n_write, n_skip, args.output)


if __name__ == "__main__":
    main()
