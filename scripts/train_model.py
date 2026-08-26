"""Offline entry point: train h_fire on data/training/*.jsonl and save the artefact.

CPU-only; finishes in seconds against the synthetic bootstrap set. GPU training is
explicitly out of scope for this laptop build (see DECISIONS.md) - if a GPU-backed
retraining pipeline is ever needed, run `src.model.train.train_model` in that environment
against real historical samples; the on-disk contract (`h_fire_v*.pkl` + metrics JSON) does
not change.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.train import train_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    metrics = train_model()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
