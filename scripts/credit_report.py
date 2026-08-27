"""Live credit meter (SRS NFR-6/AC-5): sums quoted Mireye credits from the JSONL logs and
reports spend against the V1 quality envelope (~300,000-400,000 credits).

    python scripts/credit_report.py [--days N]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
ENVELOPE_LOW = 300_000
ENVELOPE_HIGH = 400_000


def load_credit_records(days: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    records = []
    for path in sorted(LOG_DIR.glob("*.jsonl")):
        try:
            day = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if day < cutoff.replace(hour=0, minute=0, second=0, microsecond=0):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("kind") == "credit_usage":
                    records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Mireye credit spend against the V1 envelope")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    records = load_credit_records(args.days)
    quoted_total = sum(r.get("quoted_credits") or 0.0 for r in records)
    actual_total = sum(r.get("actual_credits") or 0.0 for r in records if r.get("actual_credits") is not None)
    n_calls = len(records)

    print(f"Credit usage over the last {args.days} day(s):")
    print(f"  calls logged:      {n_calls}")
    print(f"  quoted credits:    {quoted_total:,.0f}")
    print(f"  actual credits:    {actual_total:,.0f}")
    print(f"  V1 envelope:       {ENVELOPE_LOW:,} - {ENVELOPE_HIGH:,}")
    pct = (actual_total / ENVELOPE_LOW * 100) if ENVELOPE_LOW else 0.0
    print(f"  % of low envelope: {pct:.2f}%")
    if actual_total > ENVELOPE_HIGH:
        print("  WARNING: spend exceeds the V1 quality envelope's high end.")


if __name__ == "__main__":
    main()
