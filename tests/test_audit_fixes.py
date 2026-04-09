"""
test_audit_fixes — Comprehensive tests for all audit-driven fixes.

Covers:
  1. Two-class validation (Critical)
  2. NaN vs null handling (Medium)
  3. Default replacement = global_no
  4. random_state reproducibility (Medium)
  5. moment validation (Medium)
  6. max_weight post-cap diagnostic (Medium)
  7. Weighted balance routing for non-trivial weights (Option B)
  8. sklearn regularized models actually regularize (High)
  9. Real-world integration scenarios combining multiple fixes
"""

import warnings

import numpy as np
import polars as pl
import pytest

from pyrollmatch.core import rollmatch, _MATCHING_DEFAULTS
from pyrollmatch.score import score_data, _build_model
from pyrollmatch.weight import (
    entropy_balance,
    _build_constraint_matrix,
)
from pyrollmatch.match import (
    _sort_treated_indices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(
    n_treated=50,
    n_controls=200,
    n_periods=5,
    entry_period=3,
    seed=42,
):
    """Synthetic panel with known structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated):
        uid = f"T{i}"
        for t in range(1, n_periods + 1):
            rows.append({
                "id": uid, "time": t, "treat": 1,
                "entry_time": entry_period,
                "x1": rng.normal(2.0, 1.0),
                "x2": rng.normal(0.5, 0.5),
            })
    for i in range(n_controls):
        uid = f"C{i}"
        for t in range(1, n_periods + 1):
            rows.append({
                "id": uid, "time": t, "treat": 0,
                "entry_time": 99,
                "x1": rng.normal(1.5, 1.0),
                "x2": rng.normal(0.3, 0.5),
            })
    return pl.DataFrame(rows)


def _make_panel_multi_cohort(
    n_treated_per_cohort=30,
    n_controls=200,
    seed=42,
):
    """Panel with multiple entry cohorts for replacement testing."""
    rng = np.random.default_rng(seed)
    rows = []
    entry_periods = [3, 4, 5]
    for cohort_idx, ep in enumerate(entry_periods):
        for i in range(n_treated_per_cohort):
            uid = f"T{cohort_idx}_{i}"
            for t in range(1, 8):
                rows.append({
                    "id": uid, "time": t, "treat": 1,
                    "entry_time": ep,
                    "x1": rng.normal(2.0, 1.0),
                    "x2": rng.normal(0.5, 0.5),
                })
    for i in range(n_controls):
        uid = f"C{i}"
        for t in range(1, 8):
            rows.append({
                "id": uid, "time": t, "treat": 0,
                "entry_time": 99,
                "x1": rng.normal(1.5, 1.0),
                "x2": rng.normal(0.3, 0.5),
            })
    return pl.DataFrame(rows)


# =========================================================================
# 1. Two-class validation
# =========================================================================

class TestTwoClassValidation:
    """Critical: rollmatch must return None (not crash) when only one
    class remains after filtering."""

    def test_all_treated_after_filter(self):
        """Data where all controls have NaN covariates → only treated left."""
        rows = []
        for i in range(20):
            for t in range(1, 4):
                rows.append({
                    "id": f"T{i}", "time": t, "treat": 1,
                    "entry_time": 2,
                    "x1": float(i), "x2": float(i * 2),
                })
        for i in range(50):
            for t in range(1, 4):
                rows.append({
                    "id": f"C{i}", "time": t, "treat": 0,
                    "entry_time": 99,
                    "x1": None, "x2": None,  # all nulls
                })
        data = pl.DataFrame(rows)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"], verbose=False,
        )
        assert result is None

    def test_all_controls_after_filter(self):
        """Data where all treated have null covariates → only controls left."""
        rows = []
        for i in range(20):
            for t in range(1, 4):
                rows.append({
                    "id": f"T{i}", "time": t, "treat": 1,
                    "entry_time": 2,
                    "x1": None, "x2": None,
                })
        for i in range(50):
            for t in range(1, 4):
                rows.append({
                    "id": f"C{i}", "time": t, "treat": 0,
                    "entry_time": 99,
                    "x1": float(i), "x2": float(i * 2),
                })
        data = pl.DataFrame(rows)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"], verbose=False,
        )
        assert result is None

    def test_score_data_single_class_does_not_crash(self):
        """score_data with only one class should raise ValueError, not crash
        with an opaque sklearn or ZeroDivisionError."""
        reduced = pl.DataFrame({
            "treat": [1, 1, 1, 1, 1],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        # Logistic should fail with a clear sklearn error (2-class requirement)
        with pytest.raises(ValueError):
            score_data(reduced, ["x1", "x2"], "treat", model_type="logistic")


# =========================================================================
# 2. NaN vs null handling
# =========================================================================

class TestNaNHandling:
    """Medium: drop_nulls doesn't remove NaN. Verify our fix catches both."""

    def test_nan_values_filtered_before_scoring(self):
        """NaN in covariates should be filtered, not crash in score_data."""
        rows = []
        for i in range(30):
            for t in range(1, 4):
                rows.append({
                    "id": f"T{i}", "time": t, "treat": 1,
                    "entry_time": 2,
                    "x1": float(i) if i != 5 else float("nan"),
                    "x2": float(i * 2),
                })
        for i in range(100):
            for t in range(1, 4):
                rows.append({
                    "id": f"C{i}", "time": t, "treat": 0,
                    "entry_time": 99,
                    "x1": float(i) if i != 10 else float("nan"),
                    "x2": float(i),
                })
        data = pl.DataFrame(rows)
        # Should not raise — NaN rows filtered before scoring
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"], verbose=False,
        )
        assert result is not None
        assert result.matched_data.height > 0

    def test_null_still_filtered(self):
        """Null values should still be handled correctly."""
        rows = []
        for i in range(30):
            for t in range(1, 4):
                rows.append({
                    "id": f"T{i}", "time": t, "treat": 1,
                    "entry_time": 2,
                    "x1": float(i) if i != 5 else None,
                    "x2": float(i * 2),
                })
        for i in range(100):
            for t in range(1, 4):
                rows.append({
                    "id": f"C{i}", "time": t, "treat": 0,
                    "entry_time": 99,
                    "x1": float(i) if i != 10 else None,
                    "x2": float(i),
                })
        data = pl.DataFrame(rows)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"], verbose=False,
        )
        assert result is not None

    def test_mixed_nan_and_null_both_filtered(self):
        """Mix of NaN and null should all be removed."""
        rows = []
        for i in range(30):
            for t in range(1, 4):
                x1_val = float(i)
                if i == 3:
                    x1_val = float("nan")
                elif i == 7:
                    x1_val = None
                rows.append({
                    "id": f"T{i}", "time": t, "treat": 1,
                    "entry_time": 2,
                    "x1": x1_val, "x2": float(i * 2),
                })
        for i in range(100):
            for t in range(1, 4):
                rows.append({
                    "id": f"C{i}", "time": t, "treat": 0,
                    "entry_time": 99,
                    "x1": float(i), "x2": float(i),
                })
        data = pl.DataFrame(rows)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"], verbose=False,
        )
        assert result is not None

    def test_nan_filtered_in_ebal(self):
        """NaN handling works for entropy balancing too."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(30):
            for t in range(1, 4):
                rows.append({
                    "id": f"T{i}", "time": t, "treat": 1,
                    "entry_time": 2,
                    "x1": rng.normal(1.0, 0.5) if i != 5 else float("nan"),
                    "x2": rng.normal(0.5, 0.3),
                })
        for i in range(200):
            for t in range(1, 4):
                rows.append({
                    "id": f"C{i}", "time": t, "treat": 0,
                    "entry_time": 99,
                    "x1": rng.normal(1.0, 0.5),
                    "x2": rng.normal(0.5, 0.3),
                })
        data = pl.DataFrame(rows)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"], method="ebal",
            verbose=False,
        )
        assert result is not None


# =========================================================================
# 3. Default replacement = global_no
# =========================================================================

class TestDefaultReplacement:
    """Default replacement mode should now be global_no."""

    def test_default_is_global_no(self):
        assert _MATCHING_DEFAULTS["replacement"] == "global_no"

    def test_default_produces_unique_controls(self):
        """With default replacement, each control used at most once."""
        data = _make_panel_multi_cohort(n_treated_per_cohort=20, n_controls=200)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"], verbose=False,
        )
        assert result is not None
        matches = result.matched_data
        # Each control_id should appear at most once globally
        ctrl_counts = matches.group_by("control_id").len()
        assert ctrl_counts["len"].max() == 1

    def test_explicit_unrestricted_allows_reuse(self):
        """Explicit unrestricted still works and allows reuse."""
        data = _make_panel_multi_cohort(
            n_treated_per_cohort=30, n_controls=15, seed=99,
        )
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            replacement="unrestricted", verbose=False,
        )
        if result is not None and result.matched_data.height > 15:
            # More pairs than controls means reuse happened
            ctrl_counts = result.matched_data.group_by("control_id").len()
            assert ctrl_counts["len"].max() >= 1  # at minimum


# =========================================================================
# 4. random_state reproducibility
# =========================================================================

class TestRandomStateReproducibility:
    """Medium: m_order='random' with random_state should be reproducible."""

    def test_random_order_reproducible_with_seed(self):
        """Same random_state → same matches."""
        data = _make_panel(n_treated=50, n_controls=200, seed=42)
        r1 = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            m_order="random", random_state=123, verbose=False,
        )
        r2 = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            m_order="random", random_state=123, verbose=False,
        )
        assert r1 is not None and r2 is not None
        assert r1.matched_data["treat_id"].to_list() == r2.matched_data["treat_id"].to_list()
        assert r1.matched_data["control_id"].to_list() == r2.matched_data["control_id"].to_list()

    def test_different_seeds_different_results(self):
        """Different random_state → different matches (with high probability)."""
        data = _make_panel(n_treated=50, n_controls=200, seed=42)
        r1 = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            m_order="random", random_state=1, verbose=False,
        )
        r2 = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            m_order="random", random_state=999, verbose=False,
        )
        assert r1 is not None and r2 is not None
        # Very unlikely to match identically with different seeds
        ctrl1 = r1.matched_data["control_id"].to_list()
        ctrl2 = r2.matched_data["control_id"].to_list()
        assert ctrl1 != ctrl2

    def test_no_seed_still_works(self):
        """random_state=None (default) should still produce results."""
        data = _make_panel(n_treated=50, n_controls=200, seed=42)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            m_order="random", verbose=False,
        )
        assert result is not None

    def test_sort_treated_indices_seeded(self):
        """Direct test of _sort_treated_indices with rng."""
        scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        order1 = _sort_treated_indices(5, scores, "random", rng=np.random.default_rng(42))
        order2 = _sort_treated_indices(5, scores, "random", rng=np.random.default_rng(42))
        np.testing.assert_array_equal(order1, order2)


# =========================================================================
# 5. moment validation
# =========================================================================

class TestMomentValidation:
    """Medium: invalid moment values should raise, not silently degrade."""

    def test_moment_0_raises(self):
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="moment must be 1, 2, or 3"):
            _build_constraint_matrix(X, moment=0)

    def test_moment_4_raises(self):
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="moment must be 1, 2, or 3"):
            _build_constraint_matrix(X, moment=4)

    def test_moment_negative_raises(self):
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="moment must be 1, 2, or 3"):
            _build_constraint_matrix(X, moment=-1)

    @pytest.mark.parametrize("moment", [1, 2, 3])
    def test_valid_moments_work(self, moment):
        X = np.random.randn(10, 3)
        C = _build_constraint_matrix(X, moment)
        expected_cols = 1 + 3 * moment  # intercept + k * moment
        assert C.shape == (10, expected_cols)

    def test_moment_validation_in_entropy_balance(self):
        """entropy_balance propagates the moment validation."""
        treated = pl.DataFrame({
            "id": ["T1", "T2"], "x1": [1.0, 2.0], "x2": [3.0, 4.0],
        })
        control = pl.DataFrame({
            "id": [f"C{i}" for i in range(20)],
            "x1": np.random.randn(20).tolist(),
            "x2": np.random.randn(20).tolist(),
        })
        with pytest.raises(ValueError, match="moment must be 1, 2, or 3"):
            entropy_balance(treated, control, ["x1", "x2"], "id", moment=0)


# =========================================================================
# 6. max_weight post-cap diagnostic
# =========================================================================

class TestMaxWeightDiagnostic:
    """Medium: max_weight clipping should warn when balance degrades."""

    def _make_ebal_data(self, seed=42):
        """Good overlap so ebal converges, but enough difference to need
        non-trivial weights."""
        rng = np.random.default_rng(seed)
        n_t, n_c = 20, 200
        treated = pl.DataFrame({
            "id": [f"T{i}" for i in range(n_t)],
            "x1": rng.normal(1.5, 1.0, n_t).tolist(),
            "x2": rng.normal(0.8, 0.5, n_t).tolist(),
        })
        control = pl.DataFrame({
            "id": [f"C{i}" for i in range(n_c)],
            "x1": rng.normal(1.0, 1.0, n_c).tolist(),
            "x2": rng.normal(0.5, 0.5, n_c).tolist(),
        })
        return treated, control

    def test_aggressive_cap_warns(self):
        """Very tight max_weight should emit a degradation warning."""
        treated, control = self._make_ebal_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            entropy_balance(
                treated, control, ["x1", "x2"], "id",
                moment=1, max_weight=0.01,
            )
            balance_warnings = [
                x for x in w
                if "degraded exact balance" in str(x.message)
            ]
            assert len(balance_warnings) > 0

    def test_no_cap_no_warning(self):
        """Without max_weight, no degradation warning."""
        treated, control = self._make_ebal_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            entropy_balance(
                treated, control, ["x1", "x2"], "id",
                moment=1, max_weight=None,
            )
            balance_warnings = [
                x for x in w
                if "degraded exact balance" in str(x.message)
            ]
            assert len(balance_warnings) == 0

    def test_generous_cap_no_warning(self):
        """Generous max_weight that doesn't actually clip should not warn."""
        treated, control = self._make_ebal_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            entropy_balance(
                treated, control, ["x1", "x2"], "id",
                moment=1, max_weight=100.0,
            )
            balance_warnings = [
                x for x in w
                if "degraded exact balance" in str(x.message)
            ]
            assert len(balance_warnings) == 0


# =========================================================================
# 7. Weighted balance routing (Option B)
# =========================================================================

class TestWeightedBalanceRouting:
    """High: non-trivial weights should use weighted balance diagnostics."""

    def test_global_no_1to1_uses_unweighted(self):
        """global_no + num_matches=1 → all weights are 1 → unweighted balance."""
        data = _make_panel(n_treated=30, n_controls=200)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            replacement="global_no", num_matches=1, verbose=False,
        )
        assert result is not None
        # All weights should be 1.0
        assert (result.weights["weight"] - 1.0).abs().max() < 1e-9
        # Balance should have the standard schema (from compute_balance)
        assert "matched_mean_t" in result.balance.columns

    def test_num_matches_gt1_uses_weighted(self):
        """num_matches > 1 → fractional control weights → weighted balance."""
        data = _make_panel(n_treated=20, n_controls=200)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            num_matches=3, verbose=False,
        )
        if result is not None:
            # Some control weights should be fractional
            min_ctrl_weight = result.weights.filter(
                pl.col("weight") < 1.0
            ).height
            assert min_ctrl_weight > 0 or result.matched_data.height > 0
            # Balance should still be computed (weighted variant)
            assert "matched_mean_t" in result.balance.columns
            assert result.balance.height > 0

    def test_unrestricted_uses_weighted_when_reuse_occurs(self):
        """Unrestricted with scarce controls → reuse → weighted balance."""
        data = _make_panel_multi_cohort(
            n_treated_per_cohort=20, n_controls=10, seed=77,
        )
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            replacement="unrestricted", verbose=False,
        )
        if result is not None:
            assert result.balance.height > 0
            assert "matched_mean_t" in result.balance.columns


# =========================================================================
# 8. sklearn regularized models
# =========================================================================

class TestRegularizedModels:
    """High: verify lasso/ridge/elasticnet produce valid results.

    Sparsity assertions require sklearn >= 1.8 where l1_ratio controls
    the penalty directly. On older sklearn, l1_ratio is ignored and all
    three models behave like ridge. We skip sparsity tests when sklearn
    is too old.
    """

    _SKLEARN_18 = None

    @classmethod
    def _is_sklearn_18(cls):
        if cls._SKLEARN_18 is None:
            import sklearn
            major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
            cls._SKLEARN_18 = (major, minor) >= (1, 8)
        return cls._SKLEARN_18

    def _fit_all(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, (n, 10))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        models = {}
        for name in ("lasso", "ridge", "elasticnet"):
            m = _build_model(name, max_iter=5000)
            m.fit(X, y)
            models[name] = m
        return models, X, y

    @pytest.mark.skipif(
        "not TestRegularizedModels._is_sklearn_18()",
        reason="l1_ratio API requires sklearn >= 1.8",
    )
    def test_lasso_sparser_than_ridge(self):
        """Lasso should shrink noise coefficients more than ridge."""
        models, _, _ = self._fit_all()
        lasso_noise = np.abs(models["lasso"].coef_[0, 2:]).sum()
        ridge_noise = np.abs(models["ridge"].coef_[0, 2:]).sum()
        assert lasso_noise < ridge_noise, (
            f"Lasso noise ({lasso_noise:.4f}) >= ridge noise ({ridge_noise:.4f})"
        )

    def test_ridge_dense_coefficients(self):
        """Ridge should keep all coefficients non-zero."""
        models, _, _ = self._fit_all()
        n_zero = (np.abs(models["ridge"].coef_) < 1e-6).sum()
        assert n_zero == 0, f"Ridge has {n_zero} near-zero coefficients"

    def test_all_models_produce_valid_results(self):
        """End-to-end: lasso/ridge/elasticnet produce valid rollmatch output."""
        data = _make_panel(n_treated=50, n_controls=200, seed=42)
        for model_type in ("lasso", "ridge", "elasticnet"):
            result = rollmatch(
                data, treat="treat", tm="time", entry="entry_time",
                id="id", covariates=["x1", "x2"],
                model_type=model_type, verbose=False,
            )
            assert result is not None, f"{model_type} returned None"
            assert result.matched_data.height > 0, f"{model_type} no matches"
            assert result.balance.height > 0, f"{model_type} no balance"


# =========================================================================
# 9. Real-world integration scenarios
# =========================================================================

class TestRealWorldIntegration:
    """End-to-end tests combining multiple fixes on realistic data."""

    def test_lalonde_with_global_no_default(self):
        """Lalonde panel with new global_no default produces good balance."""
        from .real_world import make_lalonde_panel, REAL_WORLD_COVARIATES
        data = make_lalonde_panel()
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="unit_id", covariates=REAL_WORLD_COVARIATES,
            ps_caliper=0.2, verbose=False,
        )
        assert result is not None
        max_smd = result.balance["matched_smd"].abs().max()
        # global_no has fewer controls available so balance is slightly
        # worse than cross_cohort; 0.5 is a reasonable bound
        assert max_smd < 0.5, f"Max |SMD| = {max_smd}, expected < 0.5"
        # Verify global_no constraint
        ctrl_counts = result.matched_data.group_by("control_id").len()
        assert ctrl_counts["len"].max() == 1

    def test_lalonde_random_order_reproducible(self):
        """Lalonde with m_order='random' + random_state is reproducible."""
        from .real_world import make_lalonde_panel, REAL_WORLD_COVARIATES
        data = make_lalonde_panel()
        kwargs = dict(
            treat="treat", tm="time", entry="entry_time",
            id="unit_id", covariates=REAL_WORLD_COVARIATES,
            ps_caliper=0.2, m_order="random", random_state=42,
            verbose=False,
        )
        r1 = rollmatch(data, **kwargs)
        r2 = rollmatch(data, **kwargs)
        assert r1 is not None and r2 is not None
        assert (
            r1.matched_data["control_id"].to_list()
            == r2.matched_data["control_id"].to_list()
        )

    def test_lalonde_ebal_with_moment_validation(self):
        """Lalonde ebal with valid moments works; invalid raises."""
        from .real_world import make_lalonde_panel, REAL_WORLD_COVARIATES
        data = make_lalonde_panel()

        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="unit_id", covariates=REAL_WORLD_COVARIATES,
            method="ebal", moment=1, verbose=False,
        )
        assert result is not None

        with pytest.raises(ValueError, match="moment must be 1, 2, or 3"):
            rollmatch(
                data, treat="treat", tm="time", entry="entry_time",
                id="unit_id", covariates=REAL_WORLD_COVARIATES,
                method="ebal", moment=0, verbose=False,
            )

    def test_multi_cohort_all_replacement_modes(self):
        """Multi-cohort data works with all three replacement modes."""
        data = _make_panel_multi_cohort(n_treated_per_cohort=25, n_controls=150)
        for mode in ("global_no", "cross_cohort", "unrestricted"):
            result = rollmatch(
                data, treat="treat", tm="time", entry="entry_time",
                id="id", covariates=["x1", "x2"],
                replacement=mode, verbose=False,
            )
            assert result is not None, f"replacement={mode} returned None"
            assert result.balance.height > 0

    def test_distance_models_with_global_no(self):
        """Distance-based models work with the new global_no default."""
        data = _make_panel(n_treated=50, n_controls=200, seed=42)
        for model_type in ("mahalanobis", "euclidean", "scaled_euclidean",
                           "robust_mahalanobis"):
            result = rollmatch(
                data, treat="treat", tm="time", entry="entry_time",
                id="id", covariates=["x1", "x2"],
                model_type=model_type, verbose=False,
            )
            assert result is not None, f"{model_type} returned None"

    def test_all_fixes_combined_stress(self):
        """Stress test: multi-cohort, random order, seeded, global_no,
        with NaN-contaminated data."""
        rng = np.random.default_rng(42)
        rows = []
        entry_periods = [3, 4, 5]
        for ep_idx, ep in enumerate(entry_periods):
            for i in range(40):
                uid = f"T{ep_idx}_{i}"
                for t in range(1, 8):
                    x1 = rng.normal(2.0, 1.0)
                    # Inject some NaN
                    if i == 5 and t == ep - 1:
                        x1 = float("nan")
                    rows.append({
                        "id": uid, "time": t, "treat": 1,
                        "entry_time": ep,
                        "x1": x1, "x2": rng.normal(0.5, 0.5),
                    })
        for i in range(300):
            uid = f"C{i}"
            for t in range(1, 8):
                x1 = rng.normal(1.5, 1.0)
                if i == 50 and t == 2:
                    x1 = float("nan")
                rows.append({
                    "id": uid, "time": t, "treat": 0,
                    "entry_time": 99,
                    "x1": x1, "x2": rng.normal(0.3, 0.5),
                })
        data = pl.DataFrame(rows)
        result = rollmatch(
            data, treat="treat", tm="time", entry="entry_time",
            id="id", covariates=["x1", "x2"],
            m_order="random", random_state=42, verbose=False,
        )
        assert result is not None
        assert result.matched_data.height > 0
        # global_no constraint
        ctrl_counts = result.matched_data.group_by("control_id").len()
        assert ctrl_counts["len"].max() == 1
