"""Job C pipeline stages. Huygens is never the claim.

    python scripts/run_job_c.py --stage tapes --resume
    python scripts/run_job_c.py --stage points --resume
    python scripts/run_job_c.py --stage elmfire --resume
    python scripts/run_job_c.py --stage w --resume
    python scripts/run_job_c.py --stage eval
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STAGES = {
    "inventory": "scripts/inventory_wfigs_daily_tapes.py",
    "tapes": "scripts/download_job_c_tapes.py",
    "points": "scripts/build_job_c_book.py",
    "elmfire": "scripts/enrich_job_c_elmfire.py",
    "w": "scripts/encode_job_c_w.py",
    "eval": "scripts/evaluate_job_c.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGES), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-fires", type=int, default=None)
    args, extra = parser.parse_known_args()
    script = ROOT / STAGES[args.stage]
    sys.argv = [str(script)]
    if args.resume and args.stage in {"tapes", "points", "elmfire", "w"}:
        sys.argv.append("--resume")
    if args.max_fires is not None and args.stage in {"tapes", "points", "elmfire"}:
        sys.argv.extend(["--max-fires", str(args.max_fires)])
    sys.argv.extend(extra)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
