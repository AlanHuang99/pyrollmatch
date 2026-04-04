"""Real-world test helpers based on MatchIt's Lalonde example dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shlex

import polars as pl


LALONDE_DATA_PATH = Path(__file__).resolve().parent / "data" / "lalonde.tab"
LALONDE_TREATED_COUNT = 185
LALONDE_CONTROL_COUNT = 429
VALID_COHORT_STRATEGIES = ("age_split", "hash_split")
REAL_WORLD_COVARIATES = [
    "age",
    "educ",
    "married",
    "nodegree",
    "race_black",
    "race_hispan",
    "re74_k",
    "re75_k",
]


def load_lalonde() -> pl.DataFrame:
    """Load the Lalonde dataset vendored from MatchIt."""
    rows = []
    with LALONDE_DATA_PATH.open(encoding="utf-8") as handle:
        next(handle)  # header omits the row-name/id column present in the data rows
        for line in handle:
            if not line.strip():
                continue
            parts = shlex.split(line)
            rows.append({
                "unit_id": parts[0],
                "treat": int(parts[1]),
                "age": int(parts[2]),
                "educ": int(parts[3]),
                "race": parts[4],
                "married": int(parts[5]),
                "nodegree": int(parts[6]),
                "re74": float(parts[7]),
                "re75": float(parts[8]),
                "re78": float(parts[9]),
            })

    return (
        pl.DataFrame(rows)
        .with_columns([
            (pl.col("race") == "black").cast(pl.Int8).alias("race_black"),
            (pl.col("race") == "hispan").cast(pl.Int8).alias("race_hispan"),
            (pl.col("re74") / 1000.0).alias("re74_k"),
            (pl.col("re75") / 1000.0).alias("re75_k"),
        ])
        .sort("unit_id")
    )


def _treated_entry_time(unit_id: str, age: int, age_cutoff: float, cohort_strategy: str) -> int:
    """Assign a pseudo entry cohort to a treated unit."""
    if cohort_strategy == "age_split":
        return 3 if age <= age_cutoff else 4
    if cohort_strategy == "hash_split":
        digest = hashlib.sha256(unit_id.encode("utf-8")).hexdigest()
        return 3 if int(digest[:8], 16) % 2 == 0 else 4
    raise ValueError(
        f"cohort_strategy must be one of {VALID_COHORT_STRATEGIES}, "
        f"got {cohort_strategy!r}"
    )


def make_lalonde_panel(
    repetitions: int = 1,
    cohort_strategy: str = "age_split",
) -> pl.DataFrame:
    """Turn Lalonde into a simple two-cohort rolling-entry panel.

    Treated units enter in period 3 or 4. Two deterministic cohort
    assignments are supported:

    - ``"age_split"``: split by treated-group median age. This is a
      stress-test configuration because cohort membership is correlated
      with a matching covariate.
    - ``"hash_split"``: split by stable hash of ``unit_id``. This is a
      more neutral cohort assignment for validation sweeps.

    Controls never enter (sentinel entry 99). Earnings are scaled to
    thousands to keep logistic fitting numerically stable in larger
    replicated pressure tests.
    """
    base = load_lalonde()
    base_rows = list(base.iter_rows(named=True))
    treated_age_cutoff = base.filter(pl.col("treat") == 1)["age"].median()

    rows = []
    for rep in range(repetitions):
        income_scale = 1.0 + rep * 0.01
        age_shift = rep % 3
        for row in base_rows:
            entry_time = 99 if row["treat"] == 0 else _treated_entry_time(
                row["unit_id"], row["age"], treated_age_cutoff, cohort_strategy,
            )
            unit_id = row["unit_id"] if repetitions == 1 else f"{row['unit_id']}_rep{rep}"
            for time_period in range(1, 5):
                rows.append({
                    "unit_id": unit_id,
                    "time": time_period,
                    "treat": row["treat"],
                    "entry_time": entry_time,
                    "age": row["age"] + age_shift + (time_period - 1),
                    "educ": row["educ"],
                    "married": row["married"],
                    "nodegree": row["nodegree"],
                    "race_black": row["race_black"],
                    "race_hispan": row["race_hispan"],
                    "re74_k": row["re74_k"] * income_scale,
                    "re75_k": row["re75_k"] * income_scale,
                })

    return pl.DataFrame(rows)
