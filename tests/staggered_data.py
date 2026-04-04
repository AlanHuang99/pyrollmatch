"""Helpers for the large staggered parquet test fixture."""

from __future__ import annotations

from pathlib import Path
import json

import polars as pl


DATA_DIR = Path(__file__).resolve().parent / "data"
STAGGERED_PANEL_LARGE_PATH = DATA_DIR / "staggered_panel_large.parquet"
STAGGERED_PANEL_LARGE_METADATA_PATH = DATA_DIR / "staggered_panel_large_metadata.json"
STAGGERED_PANEL_LARGE_COVARIATES = [
    "risk_score",
    "size_index",
    "engagement",
    "growth_index",
    "stability_index",
    "cost_index",
    "support_need",
    "activity_index",
]


def load_staggered_panel_large() -> pl.DataFrame:
    """Load the large staggered parquet fixture."""
    return pl.read_parquet(STAGGERED_PANEL_LARGE_PATH)


def load_staggered_panel_large_metadata() -> dict:
    """Load metadata for the large staggered parquet fixture."""
    return json.loads(STAGGERED_PANEL_LARGE_METADATA_PATH.read_text(encoding="utf-8"))
