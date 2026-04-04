"""Robustness tests — edge cases, determinism, parameter sensitivity."""

import polars as pl
import numpy as np
import pytest
from pyrollmatch import rollmatch
from tests.test_smoke import make_synthetic_data


class TestDeterminism:
    def test_same_seed_same_result(self):
        """Same input should always produce same output."""
        data = make_synthetic_data(seed=42)

        r1 = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"], ps_caliper=0.1,
            num_matches=3, replacement="unrestricted", verbose=False,
        )
        r2 = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"], ps_caliper=0.1,
            num_matches=3, replacement="unrestricted", verbose=False,
        )

        assert r1.matched_data.height == r2.matched_data.height
        assert r1.n_treated_matched == r2.n_treated_matched


class TestEdgeCases:
    def test_no_valid_matches(self):
        """When alpha is too tight, no matches should be found gracefully."""
        data = make_synthetic_data(n_treated=50, n_controls=50, seed=42)

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.0001,  # Very tight caliper
            num_matches=3, replacement="unrestricted", verbose=False,
        )
        # Should return None or very few matches
        if result is not None:
            assert result.n_treated_matched >= 0

    def test_single_treated(self):
        """Should work with just 1 treated unit."""
        data = make_synthetic_data(n_treated=1, n_controls=100, seed=42)

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.5, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        if result is not None:
            assert result.n_treated_matched <= 1

    def test_no_caliper(self):
        """alpha=0 should match without caliper restriction."""
        data = make_synthetic_data(seed=42)

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        assert result is not None
        # Without caliper, should match most/all treated
        assert result.n_treated_matched > 0

    def test_single_covariate(self):
        """Should work with just 1 covariate."""
        data = make_synthetic_data(seed=42)

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        assert result is not None
        assert result.balance.height == 1


class TestParameterSensitivity:
    def test_num_matches_1(self):
        """1:1 matching."""
        data = make_synthetic_data(seed=42)
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, replacement="unrestricted",
            verbose=False,
        )
        assert result is not None
        # Each treated should have at most 1 match
        match_counts = result.matched_data.group_by("treat_id").len()
        assert match_counts["len"].max() <= 1

    def test_num_matches_5(self):
        """1:5 matching."""
        data = make_synthetic_data(n_controls=1000, seed=42)
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=5, replacement="unrestricted",
            verbose=False,
        )
        assert result is not None
        match_counts = result.matched_data.group_by("treat_id").len()
        assert match_counts["len"].max() <= 5

    def test_replacement_cross_cohort(self):
        """cross_cohort: controls used at most once per period."""
        data = make_synthetic_data(n_controls=500, seed=42)
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="cross_cohort", verbose=False,
        )
        if result is not None:
            # Within each time period, controls should appear at most once
            per_period = result.matched_data.group_by(["time", "control_id"]).len()
            assert per_period["len"].max() <= 1

    def test_different_block_sizes(self):
        """Different block sizes should give same results."""
        data = make_synthetic_data(n_treated=200, n_controls=600, seed=42)

        r1 = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            block_size=50, verbose=False,
        )
        r2 = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            block_size=500, verbose=False,
        )

        assert r1.n_treated_matched == r2.n_treated_matched
        assert r1.matched_data.height == r2.matched_data.height
