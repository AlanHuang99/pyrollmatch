"""
Parameter validation and guard tests.

Covers:
- Public __version__ string stays in sync with pyproject.toml
- smd_table handles an empty balance DataFrame without crashing
- rollmatch rejects invalid block_size, num_matches, caliper, ps_caliper,
  and treat column values up front with a clear ValueError, rather than
  returning None or surfacing a low-level numpy/polars exception
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrollmatch
from pyrollmatch import rollmatch
from pyrollmatch.balance import smd_table


def _panel(seed: int = 17, n_treated: int = 50, n_controls: int = 150) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated):
        et = int(rng.integers(4, 8))
        for t in range(1, 11):
            rows.append({
                "unit_id": i, "time": t, "treat": 1, "entry_time": et,
                "x1": float(rng.normal() + 0.3),
                "x2": float(rng.normal()),
            })
    for i in range(n_controls):
        for t in range(1, 11):
            rows.append({
                "unit_id": n_treated + i, "time": t, "treat": 0,
                "entry_time": 999,
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
            })
    return pl.DataFrame(rows)


@pytest.fixture
def panel():
    return _panel()


class TestVersionSync:
    """Guard against __version__ drifting from pyproject.toml."""

    def test_version_string_matches_pyproject(self):
        import tomllib
        pyproj_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproj_path, "rb") as f:
            pyproj = tomllib.load(f)
        assert pyrollmatch.__version__ == pyproj["project"]["version"], (
            f"pyrollmatch.__version__={pyrollmatch.__version__!r} "
            f"does not match pyproject.toml version "
            f"{pyproj['project']['version']!r}. "
            "Update src/pyrollmatch/__init__.py when bumping the package version."
        )


class TestSmdTableEmpty:
    """smd_table must handle an empty balance DataFrame without crashing."""

    def test_smd_table_empty_balance_does_not_crash(self, capsys):
        empty = pl.DataFrame(schema={
            "covariate": pl.Utf8,
            "full_mean_t": pl.Float64, "full_mean_c": pl.Float64,
            "full_sd_t": pl.Float64, "full_sd_c": pl.Float64,
            "full_smd": pl.Float64,
            "matched_mean_t": pl.Float64, "matched_mean_c": pl.Float64,
            "matched_sd_t": pl.Float64, "matched_sd_c": pl.Float64,
            "matched_smd": pl.Float64,
        })
        # Should not raise
        smd_table(empty)
        # Should print something indicating no data
        out = capsys.readouterr().out
        assert "no" in out.lower() or "empty" in out.lower() or "0" in out


class TestBlockSizeValidation:
    """block_size must be a positive integer across all matching paths."""

    def test_block_size_zero_raises_value_error(self, panel):
        with pytest.raises(ValueError, match="block_size"):
            rollmatch(
                panel, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1", "x2"], model_type="mahalanobis",
                block_size=0, verbose=False,
            )

    def test_block_size_negative_raises_value_error(self, panel):
        with pytest.raises(ValueError, match="block_size"):
            rollmatch(
                panel, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1", "x2"], model_type="mahalanobis",
                block_size=-5, verbose=False,
            )


class TestNumMatchesValidation:
    """num_matches must be a positive integer."""

    def test_num_matches_zero_raises(self, panel):
        with pytest.raises(ValueError, match="num_matches"):
            rollmatch(
                panel, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1"], num_matches=0, verbose=False,
            )

    def test_num_matches_negative_raises(self, panel):
        with pytest.raises(ValueError, match="num_matches"):
            rollmatch(
                panel, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1"], num_matches=-1, verbose=False,
            )


class TestCaliperValidation:
    """ps_caliper and per-variable caliper must be non-negative, and
    per-variable caliper dict keys must reference existing columns."""

    def test_negative_ps_caliper_raises(self, panel):
        with pytest.raises(ValueError, match="ps_caliper"):
            rollmatch(
                panel, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1"], ps_caliper=-0.5, verbose=False,
            )

    def test_negative_per_variable_caliper_raises(self, panel):
        with pytest.raises(ValueError, match="caliper"):
            rollmatch(
                panel, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1", "x2"], caliper={"x1": -0.5}, verbose=False,
            )

    def test_caliper_dict_unknown_column_raises(self, panel):
        with pytest.raises(ValueError, match="nonexistent"):
            rollmatch(
                panel, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1", "x2"],
                caliper={"nonexistent": 0.5}, verbose=False,
            )


class TestTreatColumnValidation:
    """treat column must be numeric {0, 1}."""

    def test_treat_string_values_raises(self, panel):
        # Rebuild the panel with a string treat column
        df = panel.with_columns(
            pl.when(pl.col("treat") == 1).then(pl.lit("T"))
            .otherwise(pl.lit("C")).alias("treat")
        )
        with pytest.raises(ValueError, match="treat"):
            rollmatch(
                df, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1"], verbose=False,
            )

    def test_treat_non_binary_integer_raises(self, panel):
        # treat=2 instead of 1 is a common mistake; should not silently
        # produce "no treated units"
        df = panel.with_columns(
            pl.when(pl.col("treat") == 1).then(pl.lit(2))
            .otherwise(pl.lit(0)).alias("treat")
        )
        with pytest.raises(ValueError, match="treat"):
            rollmatch(
                df, treat="treat", tm="time", entry="entry_time", id="unit_id",
                covariates=["x1"], verbose=False,
            )
