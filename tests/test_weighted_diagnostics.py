"""Tests for weighted balance and diagnostic functions."""

import polars as pl
import numpy as np
import pytest
from pyrollmatch import (
    rollmatch, reduce_data, score_data,
    compute_balance_weighted, balance_by_period_weighted,
    balance_test_weighted, equivalence_test_weighted,
)
from pyrollmatch.balance import _weighted_mean, _weighted_std
from tests.test_smoke import make_synthetic_data


@pytest.fixture
def ebal_result():
    """Run ebal and return (reduced_data, weights, result)."""
    data = make_synthetic_data(n_treated=100, n_controls=400, seed=42)
    result = rollmatch(
        data, "treat", "time", "entry_time", "unit_id",
        covariates=["x1", "x2", "x3"],
        method="ebal", moment=1, verbose=False,
    )
    reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
    reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
    return reduced, result.weights, result


class TestComputeBalanceWeighted:

    def test_uniform_weights_approx_unweighted(self):
        """Uniform weights should give approximately unweighted results."""
        data = make_synthetic_data(n_treated=50, n_controls=200, seed=42)
        reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])

        # Create uniform weights for all units
        all_ids = reduced["unit_id"].unique()
        weights = pl.DataFrame({
            "unit_id": all_ids.to_list(),
            "weight": [1.0] * all_ids.len(),
        })

        bal = compute_balance_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        assert bal.height == 3
        # Full and matched should be similar with uniform weights
        for row in bal.iter_rows(named=True):
            assert abs(row["full_mean_t"] - row["matched_mean_t"]) < 0.01
            assert abs(row["full_mean_c"] - row["matched_mean_c"]) < 0.01

    def test_output_schema(self, ebal_result):
        reduced, weights, result = ebal_result
        bal = compute_balance_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        expected_cols = {
            "covariate", "full_mean_t", "full_mean_c", "full_sd_t", "full_sd_c",
            "full_smd", "matched_mean_t", "matched_mean_c", "matched_sd_t",
            "matched_sd_c", "matched_smd",
        }
        assert expected_cols == set(bal.columns)

    def test_ebal_balance_reasonable(self, ebal_result):
        """Ebal-weighted pooled balance should be reasonable."""
        reduced, weights, result = ebal_result
        bal = compute_balance_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        max_smd = bal["matched_smd"].abs().max()
        # Pooled across cohorts can be imperfect; per-period is near-exact
        assert max_smd < 0.5, f"Ebal pooled balance too poor, got {max_smd:.4f}"


class TestBalanceByPeriodWeighted:

    def test_returns_two_dataframes(self, ebal_result):
        reduced, weights, _ = ebal_result
        agg, detail = balance_by_period_weighted(
            reduced, weights, "treat", "unit_id", "time", ["x1", "x2", "x3"]
        )
        assert isinstance(agg, pl.DataFrame)
        assert isinstance(detail, pl.DataFrame)

    def test_aggregate_shape(self, ebal_result):
        reduced, weights, _ = ebal_result
        agg, detail = balance_by_period_weighted(
            reduced, weights, "treat", "unit_id", "time", ["x1", "x2", "x3"]
        )
        assert agg.height == 3
        assert "max_abs_smd" in agg.columns


class TestBalanceTestWeighted:

    def test_produces_valid_output(self, ebal_result):
        reduced, weights, _ = ebal_result
        diag = balance_test_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        assert diag.height == 3
        assert "smd" in diag.columns
        assert "t_pvalue" in diag.columns
        assert "n_eff_treated" in diag.columns
        assert "n_eff_control" in diag.columns

    def test_p_values_valid(self, ebal_result):
        reduced, weights, _ = ebal_result
        diag = balance_test_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        for row in diag.iter_rows(named=True):
            p = row["t_pvalue"]
            if not np.isnan(p):
                assert 0 <= p <= 1, f"Invalid p-value: {p}"

    def test_no_ks_columns(self, ebal_result):
        """Weighted balance test should NOT include KS test."""
        reduced, weights, _ = ebal_result
        diag = balance_test_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        assert "ks_stat" not in diag.columns
        assert "ks_pvalue" not in diag.columns


class TestEquivalenceTestWeighted:

    def test_produces_valid_output(self, ebal_result):
        reduced, weights, _ = ebal_result
        equiv = equivalence_test_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        assert equiv.height == 3
        assert "tost_p" in equiv.columns
        assert "equivalent" in equiv.columns

    def test_tost_p_values_valid(self, ebal_result):
        reduced, weights, _ = ebal_result
        equiv = equivalence_test_weighted(
            reduced, weights, "treat", "unit_id", ["x1", "x2", "x3"]
        )
        for row in equiv.iter_rows(named=True):
            p = row["tost_p"]
            if not np.isnan(p):
                assert 0 <= p <= 1, f"Invalid TOST p-value: {p}"
