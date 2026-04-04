"""Generate a deterministic large staggered-treatment test fixture.

Outputs:
- tests/data/staggered_panel_large.parquet
- tests/data/staggered_panel_large_metadata.json
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import polars as pl


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "staggered_panel_large.parquet"
METADATA_PATH = DATA_DIR / "staggered_panel_large_metadata.json"

SEED = 20260404
N_UNITS = 6000
N_PERIODS = 18
ENTRY_PERIODS = np.arange(7, 15, dtype=np.int16)
CONTROL_ENTRY_SENTINEL = np.int16(99)
TARGET_TREATED_SHARE = 0.30

COVARIATES = [
    "risk_score",
    "size_index",
    "engagement",
    "growth_index",
    "stability_index",
    "cost_index",
    "support_need",
    "activity_index",
]


def _logit(x: np.ndarray) -> np.ndarray:
    """Stable logistic transform."""
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(score: np.ndarray, target_share: float) -> float:
    """Choose an intercept so mean logistic probability hits target_share."""
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _logit(mid + score).mean() < target_share:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_fixture() -> tuple[pl.DataFrame, dict]:
    """Build the staggered panel and metadata."""
    rng = np.random.default_rng(SEED)

    unit_id = np.arange(1, N_UNITS + 1, dtype=np.int32)
    region = rng.integers(0, 5, size=N_UNITS, dtype=np.int16)

    latent_risk = rng.normal(0.0, 1.0, N_UNITS).astype(np.float32)
    latent_size = rng.normal(0.0, 1.0, N_UNITS).astype(np.float32)
    latent_engagement = rng.normal(0.0, 1.0, N_UNITS).astype(np.float32)
    latent_growth = rng.normal(0.0, 1.0, N_UNITS).astype(np.float32)
    latent_stability = rng.normal(0.0, 1.0, N_UNITS).astype(np.float32)
    latent_cost = rng.normal(0.0, 1.0, N_UNITS).astype(np.float32)
    region_effect = np.array([-0.35, -0.1, 0.0, 0.12, 0.28], dtype=np.float32)[region]

    treat_score = (
        0.70 * latent_risk
        + 0.45 * latent_engagement
        + 0.30 * latent_growth
        - 0.40 * latent_stability
        + 0.25 * latent_cost
        + region_effect
        + rng.normal(0.0, 0.55, N_UNITS).astype(np.float32)
    )
    treat_intercept = _calibrate_intercept(treat_score, TARGET_TREATED_SHARE)
    treat_prob = _logit(treat_intercept + treat_score)
    treat_unit = rng.binomial(1, treat_prob, N_UNITS).astype(np.int8)

    entry_time = np.full(N_UNITS, CONTROL_ENTRY_SENTINEL, dtype=np.int16)
    treated_idx = np.flatnonzero(treat_unit == 1)
    hazard_score = (
        0.08 * latent_growth
        + 0.05 * latent_engagement
        - 0.04 * latent_stability
        + 0.03 * region_effect
        + rng.normal(0.0, 1.00, N_UNITS).astype(np.float32)
    )
    treated_order = treated_idx[np.argsort(-hazard_score[treated_idx])]
    for period, cohort_ids in zip(ENTRY_PERIODS, np.array_split(treated_order, len(ENTRY_PERIODS))):
        entry_time[cohort_ids] = period

    unit_idx = np.repeat(np.arange(N_UNITS), N_PERIODS)
    time = np.tile(np.arange(1, N_PERIODS + 1, dtype=np.int16), N_UNITS)
    trend = ((time.astype(np.float32) - 1.0) / (N_PERIODS - 1) - 0.5).astype(np.float32)
    season = np.sin(2.0 * np.pi * time / 6.0).astype(np.float32)
    season_cos = np.cos(2.0 * np.pi * time / 6.0).astype(np.float32)

    treat = treat_unit[unit_idx]
    entry = entry_time[unit_idx]
    treat_active = ((treat == 1) & (time >= entry)).astype(np.int8)
    event_time = np.where(treat_active == 1, time - entry, 0).astype(np.int16)

    risk_noise = rng.normal(0.0, 0.14, len(unit_idx)).astype(np.float32)
    size_noise = rng.normal(0.0, 0.12, len(unit_idx)).astype(np.float32)
    engagement_noise = rng.normal(0.0, 0.15, len(unit_idx)).astype(np.float32)
    growth_noise = rng.normal(0.0, 0.13, len(unit_idx)).astype(np.float32)
    stability_noise = rng.normal(0.0, 0.10, len(unit_idx)).astype(np.float32)
    cost_noise = rng.normal(0.0, 0.14, len(unit_idx)).astype(np.float32)
    support_noise = rng.normal(0.0, 0.12, len(unit_idx)).astype(np.float32)
    activity_noise = rng.normal(0.0, 0.16, len(unit_idx)).astype(np.float32)
    outcome_noise = rng.normal(0.0, 0.22, len(unit_idx)).astype(np.float32)

    risk_score = (
        latent_risk[unit_idx] + 0.20 * trend + 0.08 * season + risk_noise
    ).astype(np.float32)
    size_index = (
        latent_size[unit_idx] + 0.10 * trend + 0.04 * season_cos + size_noise
    ).astype(np.float32)
    engagement = (
        latent_engagement[unit_idx] + 0.18 * season + 0.06 * trend + engagement_noise
    ).astype(np.float32)
    growth_index = (
        latent_growth[unit_idx] + 0.24 * trend + 0.05 * season + growth_noise
    ).astype(np.float32)
    stability_index = (
        latent_stability[unit_idx] - 0.06 * np.abs(trend) + stability_noise
    ).astype(np.float32)
    cost_index = (
        0.55 * latent_risk[unit_idx]
        + 0.50 * latent_cost[unit_idx]
        + 0.06 * season_cos
        + 0.04 * trend
        + cost_noise
    ).astype(np.float32)
    support_need = (
        0.40 * latent_risk[unit_idx]
        - 0.45 * latent_stability[unit_idx]
        + 0.12 * trend
        + support_noise
    ).astype(np.float32)
    activity_index = (
        0.45 * latent_size[unit_idx]
        + 0.40 * latent_engagement[unit_idx]
        + 0.08 * season
        + activity_noise
    ).astype(np.float32)

    outcome = (
        2.0
        + 0.55 * risk_score
        + 0.40 * engagement
        + 0.35 * activity_index
        - 0.30 * cost_index
        + 0.20 * growth_index
        + 0.35 * trend
        + 0.10 * season
        + treat_active * (0.75 + 0.08 * event_time)
        + outcome_noise
    ).astype(np.float32)

    df = (
        pl.DataFrame({
            "unit_id": unit_id[unit_idx],
            "time": time,
            "treat": treat,
            "entry_time": entry,
            "treat_active": treat_active,
            "event_time": event_time,
            "region": region[unit_idx],
            "risk_score": risk_score,
            "size_index": size_index,
            "engagement": engagement,
            "growth_index": growth_index,
            "stability_index": stability_index,
            "cost_index": cost_index,
            "support_need": support_need,
            "activity_index": activity_index,
            "outcome": outcome,
        })
        .sort(["unit_id", "time"])
        .with_columns([
            pl.col("unit_id").cast(pl.Int32),
            pl.col("time").cast(pl.Int16),
            pl.col("treat").cast(pl.Int8),
            pl.col("entry_time").cast(pl.Int16),
            pl.col("treat_active").cast(pl.Int8),
            pl.col("event_time").cast(pl.Int16),
            pl.col("region").cast(pl.Int16),
        ])
    )

    n_treated_units = int(treat_unit.sum())
    n_control_units = int(N_UNITS - n_treated_units)
    entry_counts = (
        df.filter(pl.col("treat") == 1)
        .select("unit_id", "entry_time")
        .unique()
        .group_by("entry_time")
        .len()
        .sort("entry_time")
    )
    metadata = {
        "seed": SEED,
        "n_rows": int(df.height),
        "n_units": int(N_UNITS),
        "n_periods": int(N_PERIODS),
        "n_treated_units": n_treated_units,
        "n_control_units": n_control_units,
        "entry_periods": ENTRY_PERIODS.astype(int).tolist(),
        "control_entry_sentinel": int(CONTROL_ENTRY_SENTINEL),
        "covariates": COVARIATES,
        "columns": df.columns,
        "entry_counts_treated": [
            {"entry_time": int(row["entry_time"]), "n_units": int(row["len"])}
            for row in entry_counts.iter_rows(named=True)
        ],
    }
    return df, metadata


def main() -> None:
    """Generate the parquet fixture and metadata JSON."""
    df, metadata = build_fixture()
    df.write_parquet(OUTPUT_PATH, compression="zstd", compression_level=8)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {METADATA_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
