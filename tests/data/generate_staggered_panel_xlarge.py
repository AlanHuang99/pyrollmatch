"""Generate a deterministic extra-large staggered-treatment fixture.

Outputs:
- tests/data/staggered_panel_xlarge.parquet
- tests/data/staggered_panel_xlarge_metadata.json

This fixture is intentionally much larger than the checked-in smoke-test
fixture: 100k treated units, 500k control units, 20 periods, and 30
covariates, for 12M total panel rows. It writes Parquet in unit chunks to keep
peak memory bounded.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "staggered_panel_xlarge.parquet"
METADATA_PATH = DATA_DIR / "staggered_panel_xlarge_metadata.json"

SEED = 20260406
N_TREATED_UNITS = 100_000
N_CONTROL_UNITS = 500_000
N_UNITS = N_TREATED_UNITS + N_CONTROL_UNITS
N_PERIODS = 20
N_COVARIATES = 30
ENTRY_PERIODS = np.arange(7, 17, dtype=np.int16)
CONTROL_ENTRY_SENTINEL = np.int16(99)
CHUNK_UNITS = 25_000

COVARIATES = [f"x{i:02d}" for i in range(1, N_COVARIATES + 1)]

SCHEMA = pa.schema(
    [
        ("unit_id", pa.int32()),
        ("time", pa.int16()),
        ("treat", pa.int8()),
        ("entry_time", pa.int16()),
        ("treat_active", pa.int8()),
        ("event_time", pa.int16()),
        ("region", pa.int16()),
        *[(cov, pa.float32()) for cov in COVARIATES],
        ("outcome", pa.float32()),
    ]
)


def _build_unit_state() -> dict[str, np.ndarray]:
    """Build deterministic unit-level latent state with exact treated count."""
    rng = np.random.default_rng(SEED)

    region = rng.integers(0, 8, size=N_UNITS, dtype=np.int16)
    latent = rng.normal(0.0, 1.0, size=(N_UNITS, N_COVARIATES)).astype(np.float32)
    region_effect = np.array(
        [-0.35, -0.20, -0.08, 0.0, 0.10, 0.18, 0.26, 0.34],
        dtype=np.float32,
    )[region]

    treat_score = (
        0.70 * latent[:, 0]
        + 0.50 * latent[:, 1]
        + 0.35 * latent[:, 2]
        - 0.45 * latent[:, 3]
        + 0.30 * latent[:, 4]
        + 0.22 * latent[:, 5]
        - 0.18 * latent[:, 6]
        + region_effect
    )
    # Gumbel-top-k gives an exact treated count while retaining randomness.
    sample_score = treat_score + rng.gumbel(0.0, 1.0, size=N_UNITS).astype(np.float32)
    treated_idx = np.argpartition(-sample_score, N_TREATED_UNITS - 1)[:N_TREATED_UNITS]

    treat_unit = np.zeros(N_UNITS, dtype=np.int8)
    treat_unit[treated_idx] = 1

    entry_time = np.full(N_UNITS, CONTROL_ENTRY_SENTINEL, dtype=np.int16)
    hazard_score = (
        0.30 * latent[:, 2]
        + 0.24 * latent[:, 7]
        - 0.20 * latent[:, 8]
        + 0.16 * latent[:, 9]
        + 0.05 * region_effect
        + rng.normal(0.0, 0.75, size=N_UNITS).astype(np.float32)
    )
    treated_order = treated_idx[np.argsort(-hazard_score[treated_idx])]
    for period, cohort_ids in zip(ENTRY_PERIODS, np.array_split(treated_order, len(ENTRY_PERIODS))):
        entry_time[cohort_ids] = period

    return {
        "region": region,
        "latent": latent,
        "treat_unit": treat_unit,
        "entry_time": entry_time,
    }


def _make_chunk(
    unit_start: int,
    unit_end: int,
    state: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pa.Table:
    """Build one unit-contiguous panel chunk."""
    units = np.arange(unit_start, unit_end, dtype=np.int32)
    unit_idx = np.repeat(units, N_PERIODS)
    time_idx = np.tile(np.arange(1, N_PERIODS + 1, dtype=np.int16), len(units))

    trend = ((time_idx.astype(np.float32) - 1.0) / (N_PERIODS - 1) - 0.5).astype(np.float32)
    season = np.sin(2.0 * np.pi * time_idx / 6.0).astype(np.float32)
    season_cos = np.cos(2.0 * np.pi * time_idx / 6.0).astype(np.float32)

    treat = state["treat_unit"][unit_idx]
    entry = state["entry_time"][unit_idx]
    treat_active = ((treat == 1) & (time_idx >= entry)).astype(np.int8)
    event_time = np.where(treat_active == 1, time_idx - entry, 0).astype(np.int16)

    n_rows = len(unit_idx)
    trend_loading = np.linspace(-0.18, 0.26, N_COVARIATES, dtype=np.float32)
    season_loading = (0.08 * np.sin(np.arange(N_COVARIATES, dtype=np.float32))).astype(np.float32)
    cos_loading = (0.06 * np.cos(np.arange(N_COVARIATES, dtype=np.float32) / 2.0)).astype(np.float32)
    noise_scale = np.linspace(0.10, 0.20, N_COVARIATES, dtype=np.float32)

    cov_values = (
        state["latent"][unit_idx]
        + trend[:, None] * trend_loading[None, :]
        + season[:, None] * season_loading[None, :]
        + season_cos[:, None] * cos_loading[None, :]
        + rng.normal(0.0, noise_scale, size=(n_rows, N_COVARIATES)).astype(np.float32)
    ).astype(np.float32)

    outcome_noise = rng.normal(0.0, 0.25, size=n_rows).astype(np.float32)
    outcome = (
        2.0
        + 0.45 * cov_values[:, 0]
        + 0.35 * cov_values[:, 1]
        - 0.30 * cov_values[:, 3]
        + 0.25 * cov_values[:, 5]
        + 0.20 * cov_values[:, 10]
        - 0.15 * cov_values[:, 15]
        + 0.35 * trend
        + 0.12 * season
        + treat_active * (0.75 + 0.08 * event_time)
        + outcome_noise
    ).astype(np.float32)

    columns = {
        "unit_id": (unit_idx + 1).astype(np.int32),
        "time": time_idx,
        "treat": treat,
        "entry_time": entry,
        "treat_active": treat_active,
        "event_time": event_time,
        "region": state["region"][unit_idx],
    }
    for j, cov in enumerate(COVARIATES):
        columns[cov] = cov_values[:, j]
    columns["outcome"] = outcome

    return pa.Table.from_pydict(columns, schema=SCHEMA)


def build_metadata(state: dict[str, np.ndarray]) -> dict:
    """Build fixture metadata without reading the written Parquet file."""
    entry_counts = [
        {
            "entry_time": int(period),
            "n_units": int((state["entry_time"] == period).sum()),
        }
        for period in ENTRY_PERIODS
    ]
    return {
        "seed": SEED,
        "n_rows": int(N_UNITS * N_PERIODS),
        "n_units": int(N_UNITS),
        "n_periods": int(N_PERIODS),
        "n_treated_units": int(state["treat_unit"].sum()),
        "n_control_units": int(N_UNITS - state["treat_unit"].sum()),
        "n_covariates": int(N_COVARIATES),
        "entry_periods": ENTRY_PERIODS.astype(int).tolist(),
        "control_entry_sentinel": int(CONTROL_ENTRY_SENTINEL),
        "covariates": COVARIATES,
        "columns": [field.name for field in SCHEMA],
        "entry_counts_treated": entry_counts,
        "chunk_units": int(CHUNK_UNITS),
    }


def main() -> None:
    """Generate the Parquet fixture and metadata JSON."""
    start = time.perf_counter()
    state = _build_unit_state()

    OUTPUT_PATH.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        OUTPUT_PATH,
        SCHEMA,
        compression="zstd",
        compression_level=8,
        use_dictionary=False,
    )
    try:
        chunk_rng = np.random.default_rng(SEED + 1)
        for unit_start in range(0, N_UNITS, CHUNK_UNITS):
            unit_end = min(unit_start + CHUNK_UNITS, N_UNITS)
            table = _make_chunk(unit_start, unit_end, state, chunk_rng)
            writer.write_table(table)
            rows_written = unit_end * N_PERIODS
            print(
                f"wrote units {unit_start + 1:,}-{unit_end:,} "
                f"({rows_written:,}/{N_UNITS * N_PERIODS:,} rows)",
                flush=True,
            )
    finally:
        writer.close()

    metadata = build_metadata(state)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {METADATA_PATH}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
