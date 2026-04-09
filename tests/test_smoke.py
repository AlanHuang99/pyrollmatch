"""Smoke tests — basic functionality verification."""

import polars as pl
import numpy as np
import pytest
from pyrollmatch import reduce_data, score_data, rollmatch
from pyrollmatch.diagnostics import balance_test, equivalence_test
from tests.real_world import REAL_WORLD_COVARIATES, make_lalonde_panel


def make_synthetic_data(
    n_treated: int = 100,
    n_controls: int = 300,
    n_periods: int = 10,
    entry_range: tuple = (6, 9),
    seed: int = 42,
) -> pl.DataFrame:
    """Create synthetic panel data for testing."""
    rng = np.random.default_rng(seed)

    rows = []
    # Treated units
    for i in range(n_treated):
        entry_t = rng.integers(entry_range[0], entry_range[1] + 1)
        for t in range(1, n_periods + 1):
            # Pre-treatment: baseline activity
            # Post-treatment: increased activity
            base = rng.exponential(2.0)
            boost = 1.5 if t >= entry_t else 1.0
            rows.append({
                "unit_id": i,
                "time": t,
                "treat": 1,
                "entry_time": int(entry_t),
                "x1": float(base * boost + rng.normal(0, 0.5)),
                "x2": float(rng.exponential(1.0) * boost + rng.normal(0, 0.3)),
                "x3": float(rng.poisson(3) * boost),
            })

    # Control units
    for i in range(n_controls):
        for t in range(1, n_periods + 1):
            base = rng.exponential(2.0)
            rows.append({
                "unit_id": n_treated + i,
                "time": t,
                "treat": 0,
                "entry_time": 99,
                "x1": float(base + rng.normal(0, 0.5)),
                "x2": float(rng.exponential(1.0) + rng.normal(0, 0.3)),
                "x3": float(rng.poisson(3)),
            })

    return pl.DataFrame(rows)


@pytest.fixture
def synth_data():
    return make_synthetic_data()


class TestReduceData:
    def test_basic(self, synth_data):
        result = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id", lookback=1)
        assert result.height > 0
        # Should have both treated and control rows
        assert result.filter(pl.col("treat") == 1).height > 0
        assert result.filter(pl.col("treat") == 0).height > 0

    def test_treated_at_baseline(self, synth_data):
        result = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id", lookback=1)
        # Treated units should be at entry_time - 1
        treated = result.filter(pl.col("treat") == 1)
        assert (treated["time"] == treated["entry_time"] - 1).all()

    def test_invalid_lookback(self, synth_data):
        with pytest.raises(ValueError):
            reduce_data(synth_data, "treat", "time", "entry_time", "unit_id", lookback=0)

    def test_missing_column(self, synth_data):
        with pytest.raises(ValueError):
            reduce_data(synth_data, "treat", "time", "nonexistent", "unit_id")


class TestScoreData:
    def test_adds_score_column(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat").data
        assert "score" in scored.columns
        assert scored["score"].null_count() == 0

    def test_logit_scores_unbounded(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat", match_on="logit").data
        # Logit scores can be negative or positive
        assert scored["score"].min() < 0 or scored["score"].max() > 0

    def test_pscore_bounded(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat", match_on="pscore").data
        assert scored["score"].min() >= 0
        assert scored["score"].max() <= 1

    def test_probit_model(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat",
                           model_type="probit", match_on="logit").data
        assert "score" in scored.columns
        assert scored["score"].null_count() == 0
        # Probit scores (Φ⁻¹(p)) are unbounded but typically in [-3, 3]
        assert scored["score"].min() > -10
        assert scored["score"].max() < 10

    def test_scored_result_structure(self, synth_data):
        from pyrollmatch.score import ScoredResult
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        result = score_data(reduced, ["x1", "x2", "x3"], "treat")
        assert isinstance(result, ScoredResult)
        assert "score" in result.data.columns
        assert result.model is not None
        assert result.model_type == "logistic"
        assert len(result.covariates) == 3

    @pytest.mark.parametrize("model_type", ["gbm", "rf", "lasso", "ridge", "elasticnet"])
    def test_ml_model_types(self, synth_data, model_type):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat",
                           model_type=model_type, match_on="logit").data
        assert "score" in scored.columns
        assert scored["score"].null_count() == 0
        assert scored.height == reduced.height

    @pytest.mark.parametrize(
        "model_type,solver,l1_ratio",
        [
            ("lasso", "saga", 1.0),
            ("ridge", "lbfgs", 0.0),
            ("elasticnet", "saga", 0.5),
        ],
    )
    def test_regularized_model_configuration(
        self, synth_data, model_type, solver, l1_ratio,
    ):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        result = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type=model_type, match_on="logit",
        )
        params = result.model.get_params()
        assert params["solver"] == solver
        assert params["l1_ratio"] == l1_ratio

    def test_mahalanobis_scores(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat",
                           model_type="mahalanobis").data
        assert "score" in scored.columns
        assert scored["score"].null_count() == 0
        # Mahalanobis distances are non-negative
        assert scored["score"].min() >= 0

    def test_mahalanobis_no_fitted_model(self, synth_data):
        from pyrollmatch.score import ScoredResult
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        result = score_data(reduced, ["x1", "x2", "x3"], "treat",
                           model_type="mahalanobis")
        assert isinstance(result, ScoredResult)
        assert result.model is None  # No fitted model for mahalanobis

    def test_invalid_model_type(self, synth_data):
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        with pytest.raises(ValueError, match="model_type must be one of"):
            score_data(reduced, ["x1", "x2", "x3"], "treat", model_type="xgboost")


class TestRollmatch:
    def test_basic_matching(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        assert result is not None
        assert result.matched_data.height > 0
        assert result.n_treated_matched > 0
        assert result.weights.height > 0

    def test_weights_structure(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        # Treated should have weight=1
        treated_ids = (
            result.matched_data.select("treat_id").unique()
            .rename({"treat_id": "unit_id"})
        )
        treat_weights = result.weights.join(treated_ids, on="unit_id", how="semi")
        assert (treat_weights["weight"] == 1.0).all()

    def test_balance_computed(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        assert result.balance.height == 3  # 3 covariates
        assert "matched_smd" in result.balance.columns


class TestRollmatchModelTypes:
    """End-to-end rollmatch with each scoring model."""

    @pytest.mark.parametrize("model_type", [
        "logistic", "probit", "gbm", "rf", "lasso", "ridge", "elasticnet", "mahalanobis",
    ])
    def test_rollmatch_all_models(self, synth_data, model_type):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            model_type=model_type, verbose=False,
        )
        assert result is not None, f"rollmatch returned None for model_type={model_type}"
        assert result.matched_data.height > 0
        assert result.balance.height == 3
        max_smd = result.balance["matched_smd"].abs().max()
        # All models should achieve reasonable balance on synthetic data
        assert max_smd < 0.5, f"model_type={model_type} has max|SMD|={max_smd:.4f}"


class TestDiagnostics:
    def test_balance_test(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat").data

        diag = balance_test(
            scored, result.matched_data, "treat", "unit_id", "time", ["x1", "x2", "x3"]
        )
        assert diag.height == 3
        assert "smd" in diag.columns
        assert "t_pvalue" in diag.columns

    def test_equivalence_test(self, synth_data):
        result = rollmatch(
            synth_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        reduced = reduce_data(synth_data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat").data

        equiv = equivalence_test(
            scored, result.matched_data, "treat", "unit_id", "time", ["x1", "x2", "x3"]
        )
        assert equiv.height == 3
        assert "tost_p" in equiv.columns
        assert "equivalent" in equiv.columns


class TestRealWorldSmoke:
    def test_lalonde_example_improves_balance(self):
        """Real-world example adapted from MatchIt's Lalonde dataset."""
        data = make_lalonde_panel()

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=REAL_WORLD_COVARIATES,
            ps_caliper=0.2, num_matches=1, replacement="cross_cohort",
            verbose=False,
        )

        assert result is not None
        assert result.n_treated_matched >= 160

        max_raw_smd = result.balance["full_smd"].abs().max()
        max_matched_smd = result.balance["matched_smd"].abs().max()
        assert max_matched_smd < max_raw_smd
        assert max_matched_smd < 0.15
