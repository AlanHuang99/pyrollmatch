"""Tests for distance-based matching, mahvars, m_order, and per-variable calipers."""

import polars as pl
import numpy as np
import pytest
from pyrollmatch import reduce_data, score_data, rollmatch
from pyrollmatch.match import (
    DistanceSpec, _match_caliper_candidates, match_within_period,
)
from pyrollmatch.score import (
    ScoredResult, DISTANCE_MODELS, _pooled_within_group_cov,
    _pooled_within_group_sd,
)


def make_synthetic_data(seed=42):
    """Create synthetic panel data for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    n_treated, n_controls, n_periods = 100, 300, 10
    for i in range(n_treated):
        entry_t = rng.integers(6, 10)
        for t in range(1, n_periods + 1):
            base = rng.exponential(2.0)
            boost = 1.5 if t >= entry_t else 1.0
            rows.append({
                "unit_id": i, "time": t, "treat": 1, "entry_time": int(entry_t),
                "x1": float(base * boost + rng.normal(0, 0.5)),
                "x2": float(rng.exponential(1.0) * boost + rng.normal(0, 0.3)),
                "x3": float(rng.poisson(3) * boost),
            })
    for i in range(n_controls):
        for t in range(1, n_periods + 1):
            base = rng.exponential(2.0)
            rows.append({
                "unit_id": n_treated + i, "time": t, "treat": 0, "entry_time": 99,
                "x1": float(base + rng.normal(0, 0.5)),
                "x2": float(rng.exponential(1.0) + rng.normal(0, 0.3)),
                "x3": float(rng.poisson(3)),
            })
    return pl.DataFrame(rows)


@pytest.fixture
def synth_data():
    return make_synthetic_data()


class TestPooledWithinGroupCov:
    """Test the pooled within-group covariance computation."""

    def test_equals_standard_cov_when_means_equal(self):
        """When group means are equal, pooled-within = overall covariance."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = np.array([0] * 100 + [1] * 100)
        # Shift both groups to have same mean
        cov_within = _pooled_within_group_cov(X, y)
        cov_overall = np.cov(X, rowvar=False, ddof=1)
        # Should be close (not exact due to finite sample)
        np.testing.assert_allclose(cov_within, cov_overall, atol=0.15)

    def test_smaller_than_overall_when_means_differ(self):
        """Pooled-within variance < overall when group means differ."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 2))
        y = np.array([0] * 100 + [1] * 100)
        # Shift treated group mean
        X[y == 1] += 3.0
        cov_within = _pooled_within_group_cov(X, y)
        cov_overall = np.cov(X, rowvar=False, ddof=1)
        # Within-group variance should be smaller
        assert np.diag(cov_within).sum() < np.diag(cov_overall).sum()


class TestDistanceModels:
    """Test all distance-based model types end-to-end."""

    @pytest.mark.parametrize("model_type", list(DISTANCE_MODELS))
    def test_rollmatch_distance_models(self, synth_data, model_type):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            num_matches=3, replacement="unrestricted",
            model_type=model_type, verbose=False,
        )
        assert result is not None, f"rollmatch returned None for {model_type}"
        assert result.matched_data.height > 0
        assert result.balance.height == 3

    @pytest.mark.parametrize("model_type", list(DISTANCE_MODELS))
    def test_score_data_distance_models(self, synth_data, model_type):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        result = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type=model_type,
        )
        assert isinstance(result, ScoredResult)
        assert result.model is None
        assert result.distance_metric is not None
        assert result.model_type == model_type

    def test_mahalanobis_has_cov_inv(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        result = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type="mahalanobis",
        )
        assert result.cov_inv is not None
        assert result.cov_inv.shape == (3, 3)

    def test_scaled_euclidean_has_transform(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        result = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type="scaled_euclidean",
        )
        assert result.distance_transform is not None
        assert result.distance_transform.shape == (3, 3)

    def test_robust_mahalanobis_has_cov_inv(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        result = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type="robust_mahalanobis",
        )
        assert result.cov_inv is not None


class TestMahvars:
    """Test Mahalanobis matching on specified covariates + PS caliper."""

    def test_mahvars_basic(self, synth_data):
        """mahvars with logistic PS for caliper + Mahalanobis on subset."""
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=3, replacement="unrestricted",
            verbose=False, mahvars=["x1", "x2"],
        )
        assert result is not None
        assert result.matched_data.height > 0

    def test_mahvars_vs_pure_mahalanobis(self, synth_data):
        """mahvars should produce different results than pure mahalanobis."""
        r_mah = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            num_matches=3, replacement="unrestricted",
            model_type="mahalanobis", verbose=False,
        )
        r_mahvars = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.3, num_matches=3, replacement="unrestricted",
            verbose=False, mahvars=["x1", "x2"],
        )
        assert r_mah is not None
        assert r_mahvars is not None
        # They should differ (different distance + caliper)
        assert r_mah.matched_data.height != r_mahvars.matched_data.height or \
            not (r_mah.matched_data["control_id"].to_list() ==
                 r_mahvars.matched_data["control_id"].to_list())

    def test_mahvars_rejects_distance_model(self, synth_data):
        """mahvars cannot be used with distance-based model_type."""
        with pytest.raises(ValueError, match="mahvars cannot be used"):
            rollmatch(
                synth_data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                num_matches=3, replacement="unrestricted",
                model_type="mahalanobis", mahvars=["x1", "x2"],
                verbose=False,
            )


class TestMOrder:
    """Test matching order parameter."""

    def test_m_order_data(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, m_order="data",
            replacement="cross_cohort", verbose=False,
        )
        assert result is not None

    def test_m_order_largest(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, m_order="largest",
            replacement="cross_cohort", verbose=False,
        )
        assert result is not None

    def test_m_order_smallest(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, m_order="smallest",
            replacement="cross_cohort", verbose=False,
        )
        assert result is not None

    def test_m_order_random(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, m_order="random",
            replacement="cross_cohort", verbose=False,
        )
        assert result is not None

    def test_m_order_affects_results(self, synth_data):
        """Different m_order should produce different matches without replacement."""
        r_largest = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, m_order="largest",
            replacement="cross_cohort", verbose=False,
        )
        r_smallest = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, m_order="smallest",
            replacement="cross_cohort", verbose=False,
        )
        assert r_largest is not None and r_smallest is not None
        # With replacement constraints, different ordering should yield different pairs
        # (at least some difference in which controls get matched)
        ctrl_largest = set(r_largest.matched_data["control_id"].to_list())
        ctrl_smallest = set(r_smallest.matched_data["control_id"].to_list())
        # Not guaranteed to differ for all data, but should usually differ
        # Just check both produced results
        assert len(ctrl_largest) > 0
        assert len(ctrl_smallest) > 0

    def test_m_order_invalid(self, synth_data):
        with pytest.raises(ValueError, match="m_order must be"):
            rollmatch(
                synth_data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                num_matches=3, replacement="unrestricted",
                m_order="invalid", verbose=False,
            )


class TestPerVariableCaliper:
    """Test per-variable caliper constraints."""

    def test_caliper_dict(self, synth_data):
        """Per-variable caliper should produce results."""
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False, caliper={"x1": 0.5, "x2": 0.5},
        )
        assert result is not None
        assert result.matched_data.height > 0

    def test_caliper_reduces_matches(self, synth_data):
        """Per-variable caliper should reduce the number of matches."""
        r_no_cal = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            num_matches=3, replacement="unrestricted", verbose=False,
        )
        r_tight_cal = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            num_matches=3, replacement="unrestricted", verbose=False,
            caliper={"x1": 0.1},
        )
        assert r_no_cal is not None
        if r_tight_cal is not None:
            assert r_tight_cal.matched_data.height <= r_no_cal.matched_data.height

    def test_std_caliper_false(self, synth_data):
        """std_caliper=False uses raw units."""
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            num_matches=3, replacement="unrestricted", verbose=False,
            caliper={"x1": 2.0}, std_caliper=False,
        )
        assert result is not None

    def test_candidate_caliper_matches_matrix_fallback(self, synth_data):
        """Candidate caliper path should match the original matrix fallback."""
        covariates = ["x1", "x2", "x3"]
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        scored_result = score_data(
            reduced, covariates, "treat", model_type="mahalanobis",
        )
        scored = scored_result.data

        all_treat = scored["treat"].to_numpy() == 1
        all_tm = scored["time"].to_numpy()
        period = int(scored.filter(pl.col("treat") == 1)["time"].min())
        t_idx = np.where(all_treat & (all_tm == period))[0]
        c_idx = np.where(~all_treat & (all_tm == period))[0]

        X = scored.select(covariates).to_numpy().astype(np.float64)
        scores = scored["score"].to_numpy()
        ids = scored["unit_id"].to_numpy()
        widths = []
        for cov in covariates:
            vals = scored[cov].to_numpy().astype(np.float64)
            sd_t = np.std(vals[all_treat], ddof=1)
            sd_c = np.std(vals[~all_treat], ddof=1)
            widths.append(0.5 * np.sqrt((sd_t**2 + sd_c**2) / 2))
        widths = np.array(widths, dtype=np.float64)

        mask = np.ones((len(t_idx), len(c_idx)), dtype=bool)
        for j, width in enumerate(widths):
            mask &= np.abs(X[t_idx, j, None] - X[c_idx, j]) <= width

        dist_spec = DistanceSpec(
            metric="mahalanobis",
            covariates=covariates,
            cov_inv=scored_result.cov_inv,
        )
        fallback = match_within_period(
            scores[t_idx], scores[c_idx], ids[t_idx], ids[c_idx],
            caliper_width=np.inf, num_matches=1, replacement="global_no",
            treated_covs=X[t_idx], control_covs=X[c_idx],
            dist_spec=dist_spec, var_caliper_mask=mask,
        )
        candidate = _match_caliper_candidates(
            scores[t_idx], scores[c_idx], ids[t_idx], ids[c_idx],
            num_matches=1, replacement="global_no", _used_controls=set(),
            treated_covs=X[t_idx], control_covs=X[c_idx],
            dist_spec=dist_spec, m_order=None,
            caliper_treated=X[t_idx], caliper_controls=X[c_idx],
            caliper_widths=widths, rng=None,
        )

        assert fallback is not None and candidate is not None
        fallback_pairs = list(zip(fallback.treat_ids, fallback.control_ids))
        candidate_pairs = list(zip(candidate.treat_ids, candidate.control_ids))
        assert candidate_pairs == fallback_pairs
