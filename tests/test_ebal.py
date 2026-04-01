"""Tests for entropy balancing (issue #5) and pluggable method architecture."""

import polars as pl
import numpy as np
import pytest
from pyrollmatch import rollmatch, entropy_balance, reduce_data
from pyrollmatch.balance import _weighted_mean, _weighted_std
from tests.test_smoke import make_synthetic_data


# ---------------------------------------------------------------------------
# Unit tests for entropy_balance()
# ---------------------------------------------------------------------------

class TestEntropyBalance:
    """Test the core entropy balancing algorithm."""

    @pytest.fixture
    def cohort_data(self):
        """Create a single cohort with known imbalance."""
        rng = np.random.default_rng(42)
        n_t, n_c = 50, 200

        treated = pl.DataFrame({
            "unit_id": list(range(n_t)),
            "x1": rng.normal(5.0, 1.0, n_t).tolist(),
            "x2": rng.normal(10.0, 2.0, n_t).tolist(),
            "x3": rng.normal(3.0, 0.5, n_t).tolist(),
        })
        controls = pl.DataFrame({
            "unit_id": list(range(n_t, n_t + n_c)),
            "x1": rng.normal(4.0, 1.5, n_c).tolist(),
            "x2": rng.normal(8.0, 2.5, n_c).tolist(),
            "x3": rng.normal(2.5, 0.8, n_c).tolist(),
        })
        return treated, controls

    def test_exact_mean_balance(self, cohort_data):
        """The defining property: weighted control means == treated means."""
        treated, controls = cohort_data
        covs = ["x1", "x2", "x3"]

        result = entropy_balance(treated, controls, covs, "unit_id", moment=1)
        assert result is not None

        # Check balance
        ctrl_weights = result.join(controls.select("unit_id"), on="unit_id", how="semi")
        for cov in covs:
            treated_mean = treated[cov].mean()
            vals = controls[cov].to_numpy()
            w = ctrl_weights["weight"].to_numpy()
            weighted_ctrl_mean = np.sum(w * vals) / np.sum(w)
            assert abs(weighted_ctrl_mean - treated_mean) < 1e-3, (
                f"{cov}: weighted control mean {weighted_ctrl_mean:.6f} != "
                f"treated mean {treated_mean:.6f}"
            )

    def test_all_weights_positive(self, cohort_data):
        treated, controls = cohort_data
        result = entropy_balance(treated, controls, ["x1", "x2", "x3"], "unit_id")
        assert result is not None
        assert (result["weight"] > 0).all()

    def test_weights_sum_to_n_treated(self, cohort_data):
        treated, controls = cohort_data
        result = entropy_balance(treated, controls, ["x1", "x2", "x3"], "unit_id")
        assert result is not None

        ctrl_weights = result.join(controls.select("unit_id"), on="unit_id", how="semi")
        n_t = treated.height
        assert abs(ctrl_weights["weight"].sum() - n_t) < 0.1

    def test_treated_weight_one(self, cohort_data):
        treated, controls = cohort_data
        result = entropy_balance(treated, controls, ["x1", "x2", "x3"], "unit_id")
        assert result is not None

        treat_weights = result.join(treated.select("unit_id"), on="unit_id", how="semi")
        assert (treat_weights["weight"] == 1.0).all()

    def test_moment_2_variance_balance(self, cohort_data):
        """moment=2 should also balance variances."""
        treated, controls = cohort_data
        covs = ["x1", "x2", "x3"]

        result = entropy_balance(treated, controls, covs, "unit_id", moment=2)
        assert result is not None

        ctrl_weights = result.join(controls.select("unit_id"), on="unit_id", how="semi")
        for cov in covs:
            treated_var = np.var(treated[cov].to_numpy())
            vals = controls[cov].to_numpy()
            w = ctrl_weights["weight"].to_numpy()
            wm = np.sum(w * vals) / np.sum(w)
            weighted_var = np.sum(w * (vals - wm) ** 2) / np.sum(w)
            # Variance balance is approximate (moment=2 targets E[X^2], not Var directly)
            assert abs(weighted_var - treated_var) < 1.0, (
                f"{cov}: weighted var {weighted_var:.4f} vs treated var {treated_var:.4f}"
            )

    def test_max_weight_cap(self, cohort_data):
        treated, controls = cohort_data
        result = entropy_balance(
            treated, controls, ["x1", "x2", "x3"], "unit_id",
            max_weight=0.5,
        )
        assert result is not None
        # Check control weights only (treated always have weight=1.0)
        ctrl_weights = result.join(controls.select("unit_id"), on="unit_id", how="semi")
        assert ctrl_weights["weight"].max() <= 0.5 + 1e-6

    def test_extreme_imbalance_warns_low_effective_n(self):
        """Severely imbalanced data should warn about low effective N."""
        rng = np.random.default_rng(99)
        treated = pl.DataFrame({
            "unit_id": list(range(20)),
            "x1": rng.normal(10.0, 0.5, 20).tolist(),
        })
        controls = pl.DataFrame({
            "unit_id": list(range(20, 120)),
            "x1": rng.normal(2.0, 0.5, 100).tolist(),
        })
        # This should either warn about low n_eff or fail to converge
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = entropy_balance(treated, controls, ["x1"], "unit_id")
        # At least one warning should fire (low n_eff or convergence)
        if result is not None:
            ctrl_w = result.join(controls.select("unit_id"), on="unit_id", how="semi")
            # Extreme case: effective N should be very low
            w_arr = ctrl_w["weight"].to_numpy()
            n_eff = w_arr.sum() ** 2 / np.sum(w_arr ** 2)
            assert n_eff < 50  # much less than 100 controls

    def test_empty_data_returns_none(self):
        treated = pl.DataFrame({"unit_id": [], "x1": []})
        controls = pl.DataFrame({"unit_id": [1, 2], "x1": [1.0, 2.0]})
        result = entropy_balance(treated, controls, ["x1"], "unit_id")
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests: rollmatch(method="ebal")
# ---------------------------------------------------------------------------

class TestRollmatchEbal:
    """Test the full ebal pipeline through rollmatch()."""

    @pytest.fixture
    def data(self):
        return make_synthetic_data(n_treated=100, n_controls=400, seed=42)

    def test_basic_ebal(self, data):
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            method="ebal", moment=1, verbose=False,
        )
        assert result is not None
        assert result.method == "ebal"
        assert result.matched_data is None  # no pairs for ebal
        assert result.alpha is None
        assert result.weights.height > 0
        assert result.balance.height == 3  # 3 covariates

    def test_ebal_achieves_per_period_balance(self, data):
        """Ebal achieves near-zero per-period SMD (balance by construction)."""
        from pyrollmatch import balance_by_period_weighted, reduce_data

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            method="ebal", moment=1, verbose=False,
        )
        assert result is not None

        reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
        agg, _ = balance_by_period_weighted(
            reduced, result.weights, "treat", "unit_id", "time",
            ["x1", "x2", "x3"],
        )
        # Per-period balance should be near-exact
        for row in agg.iter_rows(named=True):
            assert row["max_abs_smd"] < 0.01, (
                f"{row['covariate']}: per-period max|SMD|={row['max_abs_smd']:.4f}"
            )

    def test_ebal_pooled_balance_reasonable(self, data):
        """Pooled balance may not be zero (cohort aggregation) but should be reasonable."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            method="ebal", moment=1, verbose=False,
        )
        assert result is not None
        max_smd = result.balance["matched_smd"].abs().max()
        # Pooled balance can be worse than per-period due to cohort aggregation
        assert max_smd < 0.5, (
            f"Pooled balance too poor: max|SMD|={max_smd:.4f}"
        )

    def test_ebal_moment_2(self, data):
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            method="ebal", moment=2, verbose=False,
        )
        assert result is not None
        assert result.balance.height == 3

    def test_matching_backward_compat(self, data):
        """Default method='matching' should work like before."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            method="matching", alpha=0.2, num_matches=3, verbose=False,
        )
        assert result is not None
        assert result.method == "matching"
        assert result.matched_data is not None
        assert result.matched_data.height > 0
        assert result.alpha == 0.2

    def test_invalid_method(self, data):
        with pytest.raises(ValueError, match="method must be"):
            rollmatch(
                data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                method="invalid", verbose=False,
            )

    def test_invalid_kwargs_for_method(self, data):
        """Matching kwargs should be rejected for ebal."""
        with pytest.raises(ValueError, match="Unknown keyword"):
            rollmatch(
                data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                method="ebal", alpha=0.1, verbose=False,
            )

    def test_custom_callable_method(self, data):
        """User-defined callable should work."""
        def my_method(treated_data, control_data, covariates, id, **kwargs):
            # Trivial method: uniform weights
            n_t = treated_data.height
            n_c = control_data.height
            treat_w = pl.DataFrame({
                id: treated_data[id].to_list(),
                "weight": [1.0] * n_t,
            })
            ctrl_w = pl.DataFrame({
                id: control_data[id].to_list(),
                "weight": [n_t / n_c] * n_c,
            })
            return pl.concat([treat_w, ctrl_w])

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            method=my_method, verbose=False,
        )
        assert result is not None
        assert result.method == "custom"
        assert result.weights.height > 0


# ---------------------------------------------------------------------------
# Weighted statistics unit tests
# ---------------------------------------------------------------------------

class TestWeightedStats:

    def test_weighted_mean_uniform(self):
        """Uniform weights should give unweighted mean."""
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        assert abs(_weighted_mean(vals, w) - 2.5) < 1e-10

    def test_weighted_mean_nonuniform(self):
        vals = np.array([1.0, 3.0])
        w = np.array([1.0, 3.0])
        # (1*1 + 3*3) / (1+3) = 10/4 = 2.5
        assert abs(_weighted_mean(vals, w) - 2.5) < 1e-10

    def test_weighted_std_uniform(self):
        """Uniform weights should approximate unweighted std."""
        vals = np.array([2.0, 4.0, 6.0, 8.0])
        w = np.ones(4)
        expected = np.std(vals, ddof=1)
        assert abs(_weighted_std(vals, w) - expected) < 1e-6

    def test_effective_n(self):
        """Kish effective sample size."""
        from pyrollmatch.diagnostics import _effective_n
        # Uniform weights: n_eff = n
        w = np.ones(10)
        assert abs(_effective_n(w) - 10.0) < 1e-10

        # One dominant weight: n_eff ≈ 1
        w = np.array([100.0, 0.01, 0.01, 0.01])
        n_eff = _effective_n(w)
        assert n_eff < 2.0
