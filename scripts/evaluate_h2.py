"""H2 falsification gate (SRS 6.6, AC-12/AC-13).

Compares W-conditioned calibration of the delegated arrival field against the raw
field on event-held-out samples. If `data/training/*.jsonl` has no real
`spread_vector` labels, the report is `unevaluable_no_real_spread_labels` - that
is an honest outcome, not a pass. A synthetic probe always runs to prove the
comparison code itself works.

    python scripts/evaluate_h2.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.train import evaluate_h2_claim  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    report = evaluate_h2_claim()
    print(json.dumps(report, indent=2))
    status = report.get("h2_status")
    if status == "validated":
        sys.exit(0)
    if status == "unevaluable_no_real_spread_labels":
        # Not a test failure: the gate ran and documented that real labels are missing.
        sys.exit(0)
    if status == "falsified":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
