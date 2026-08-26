"""Generate 500 synthetic training samples so the watch loop can train and run before any
real historical fire data has been collected (SRS Step 8 / build prompt Step 8).

These labels are random (50% positive rate) and are explicitly NOT a research claim - they
only exist to exercise `train.py` and `model_infer` end to end for the Step 18 smoke test.
Real H1 validation (SRS AC-7/AC-8) requires real MTBS/WFIGS-derived labels (see README).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.e_packer import E_VECTOR_FIELD_ORDER  # noqa: E402
from src.features.w_encoder import encode_w  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "training" / "synthetic_001.jsonl"
N_SAMPLES = 500
N_EVENTS = 50
RANDOM_STATE = 42


def _synthetic_e_vector(rng: np.random.Generator) -> list[float]:
    values = {}
    values["rflag"] = float(rng.integers(0, 2))
    values["spc_day1_elevated"] = float(rng.integers(0, 2))
    values["firms_count_5km"] = float(rng.poisson(1.0))
    values["firms_count_10km"] = float(rng.poisson(2.0))
    values["firms_count_20km"] = float(rng.poisson(3.0))
    values["frp_sum_5km"] = float(max(0.0, rng.normal(20.0, 15.0)))
    values["dist_perim_m"] = float(rng.uniform(0, 50_000))
    values["acres"] = float(max(0.0, rng.normal(1000.0, 800.0)))
    values["containment_pct"] = float(rng.uniform(0.0, 1.0))
    values["hours_since_discovery"] = float(rng.uniform(0, 240))
    values["perimeter_unofficial"] = float(rng.integers(0, 2))
    values["wind_speed_ms"] = float(max(0.0, rng.normal(6.0, 3.0)))
    values["temp_c"] = float(rng.normal(28.0, 8.0))
    values["rh_pct"] = float(np.clip(rng.normal(25.0, 15.0), 1, 100))
    values["wind_ros_ellipse_dist_m"] = float(rng.uniform(-5_000, 60_000))
    return [values[name] for name in E_VECTOR_FIELD_ORDER]


def generate(n_samples: int = N_SAMPLES, n_events: int = N_EVENTS, seed: int = RANDOM_STATE) -> list[dict]:
    rng = np.random.default_rng(seed)
    w_dim = len(encode_w({}).vector)

    samples = []
    for i in range(n_samples):
        event_id = f"synthetic_event_{i % n_events:03d}"
        site_id = f"synthetic_site_{i:04d}"
        w_vector = rng.normal(0.0, 1.0, size=w_dim).tolist()
        w_mask = (rng.uniform(0, 1, size=w_dim) > 0.1).astype(float).tolist()  # ~90% present
        e_vector = _synthetic_e_vector(rng)
        y = int(rng.integers(0, 2))  # 50% positive rate, explicitly random (see module docstring)
        samples.append(
            {
                "site_id": site_id,
                "event_id": event_id,
                "t0": "2026-01-01T00:00:00Z",
                "w_vector": w_vector,
                "w_mask": w_mask,
                "e_vector": e_vector,
                "y": y,
            }
        )
    return samples


def main() -> None:
    samples = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Wrote {len(samples)} synthetic training samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
