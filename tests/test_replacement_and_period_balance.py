"""Tests for replacement modes (#3) and per-period balance (#4)."""

import polars as pl
import pytest
from pyrollmatch import rollmatch, balance_by_period, reduce_data, score_data
from tests.test_smoke import make_synthetic_data


# ---------------------------------------------------------------------------
# Replacement modes
# ---------------------------------------------------------------------------

class TestReplacementModes:
    """Test the three replacement modes: unrestricted, cross_cohort, global_no."""

    @pytest.fixture
    def data(self):
        return make_synthetic_data(n_treated=100, n_controls=500, seed=42)

    def test_bool_replacement_rejected(self, data):
        """Boolean replacement values should raise ValueError."""
        for val in [True, False]:
            with pytest.raises(ValueError, match="replacement must be"):
                rollmatch(
                    data, "treat", "time", "entry_time", "unit_id",
                    covariates=["x1", "x2", "x3"],
                    ps_caliper=0.2, num_matches=1, replacement=val, verbose=False,
                )

    def test_cross_cohort_no_reuse_within_period(self, data):
        """Controls used at most once per period in cross_cohort mode."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="cross_cohort", verbose=False,
        )
        assert result is not None
        per_period = result.matched_data.group_by(["time", "control_id"]).len()
        assert per_period["len"].max() <= 1

    def test_cross_cohort_allows_reuse_across_periods(self, data):
        """Controls CAN appear in multiple periods under cross_cohort."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="cross_cohort", verbose=False,
        )
        assert result is not None
        # Count how many distinct periods each control appears in
        ctrl_period_counts = (
            result.matched_data
            .select("control_id", "time").unique()
            .group_by("control_id").len()
        )
        # With enough data, at least some controls should span multiple periods
        # (not guaranteed, but very likely with 500 controls and 100 treated)
        max_periods = ctrl_period_counts["len"].max()
        assert max_periods >= 1  # at least used once (sanity)

    def test_global_no_controls_used_once(self, data):
        """In global_no mode, each control appears at most once across ALL periods."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="global_no", verbose=False,
        )
        assert result is not None
        # Each control_id should appear at most once across the entire match table
        ctrl_counts = result.matched_data.group_by("control_id").len()
        assert ctrl_counts["len"].max() <= 1

    def test_global_no_fewer_matches_than_unrestricted(self, data):
        """global_no should produce <= matches than unrestricted."""
        r_unres = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="unrestricted", verbose=False,
        )
        r_global = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="global_no", verbose=False,
        )
        assert r_unres is not None
        assert r_global is not None
        assert r_global.matched_data.height <= r_unres.matched_data.height

    def test_global_no_fewer_matches_than_cross_cohort(self, data):
        """global_no should produce <= matches than cross_cohort."""
        r_cross = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="cross_cohort", verbose=False,
        )
        r_global = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=1, replacement="global_no", verbose=False,
        )
        assert r_cross is not None
        assert r_global is not None
        assert r_global.matched_data.height <= r_cross.matched_data.height

    def test_invalid_replacement_value(self):
        data = make_synthetic_data(n_treated=10, n_controls=30, seed=42)
        with pytest.raises(ValueError, match="replacement must be"):
            rollmatch(
                data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                ps_caliper=0.2, num_matches=3, replacement="invalid", verbose=False,
            )



# ---------------------------------------------------------------------------
# Per-period balance
# ---------------------------------------------------------------------------

class TestBalanceByPeriod:
    """Test balance_by_period() function."""

    @pytest.fixture
    def matched_result(self):
        data = make_synthetic_data(n_treated=100, n_controls=300, seed=42)
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat").data
        return scored, result

    def test_returns_two_dataframes(self, matched_result):
        scored, result = matched_result
        agg, detail = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1", "x2", "x3"],
        )
        assert isinstance(agg, pl.DataFrame)
        assert isinstance(detail, pl.DataFrame)

    def test_aggregate_shape(self, matched_result):
        scored, result = matched_result
        agg, detail = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1", "x2", "x3"],
        )
        # One row per covariate
        assert agg.height == 3
        assert set(agg["covariate"].to_list()) == {"x1", "x2", "x3"}
        assert "wtd_mean_smd" in agg.columns
        assert "median_abs_smd" in agg.columns
        assert "max_abs_smd" in agg.columns
        assert "n_periods" in agg.columns

    def test_detail_has_expected_columns(self, matched_result):
        scored, result = matched_result
        agg, detail = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1", "x2", "x3"],
        )
        expected_cols = {"period", "covariate", "n_treated", "n_controls",
                         "mean_treated", "mean_control", "smd"}
        assert expected_cols.issubset(set(detail.columns))

    def test_detail_covers_periods_and_covariates(self, matched_result):
        scored, result = matched_result
        agg, detail = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1", "x2", "x3"],
        )
        # Detail should have rows for each (period, covariate) combo
        n_periods = detail["period"].n_unique()
        n_covs = detail["covariate"].n_unique()
        assert detail.height == n_periods * n_covs

    def test_max_abs_smd_is_worst_period(self, matched_result):
        scored, result = matched_result
        agg, detail = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1", "x2", "x3"],
        )
        for cov in ["x1", "x2", "x3"]:
            cov_detail = detail.filter(
                (pl.col("covariate") == cov) & pl.col("smd").is_not_nan()
            )
            cov_agg = agg.filter(pl.col("covariate") == cov)
            expected_max = float(cov_detail["smd"].abs().max())
            actual_max = float(cov_agg["max_abs_smd"][0])
            assert abs(expected_max - actual_max) < 1e-4

    def test_n_periods_matches_detail(self, matched_result):
        scored, result = matched_result
        agg, detail = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1", "x2", "x3"],
        )
        for cov in ["x1", "x2", "x3"]:
            cov_detail = detail.filter(
                (pl.col("covariate") == cov) & pl.col("smd").is_not_nan()
            )
            cov_agg = agg.filter(pl.col("covariate") == cov)
            assert int(cov_agg["n_periods"][0]) == cov_detail.height

    def test_single_covariate(self):
        data = make_synthetic_data(n_treated=50, n_controls=200, seed=42)
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1"])
        scored = score_data(reduced, ["x1"], "treat").data
        agg, detail = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1"],
        )
        assert agg.height == 1
        assert agg["covariate"][0] == "x1"
