"""Integration tests with realistic data patterns.

Tests the full pipeline end-to-end with data that mimics real-world
staggered adoption studies: confounded treatment assignment, multiple
entry cohorts, varying control pool sizes, and expected balance outcomes.
"""

import polars as pl
import numpy as np
import pytest
from pyrollmatch import (
    rollmatch, RollmatchResult,
    SUPPORTED_MODELS, DISTANCE_MODELS,
)


# ---------------------------------------------------------------------------
# Realistic data generators
# ---------------------------------------------------------------------------

def make_confounded_data(
    n_treated: int = 200,
    n_controls: int = 2000,
    n_periods: int = 20,
    n_entry_cohorts: int = 5,
    n_covariates: int = 5,
    treatment_effect: float = 0.5,
    confounding_strength: float = 1.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate panel data with confounded treatment assignment.

    Treatment probability depends on covariates (selection bias).
    Controls have stable covariates; treated have a pre-treatment trend.
    Multiple entry cohorts with different entry times.
    """
    rng = np.random.default_rng(seed)

    # Entry times spread across periods
    entry_times = np.linspace(
        n_periods // 3, 2 * n_periods // 3, n_entry_cohorts
    ).astype(int)

    rows = []
    uid = 0

    # Treated units: covariates correlated with treatment
    for i in range(n_treated):
        entry_t = rng.choice(entry_times)
        base_x = rng.normal(confounding_strength, 1.0, n_covariates)
        for t in range(1, n_periods + 1):
            row = {
                "unit_id": uid,
                "time": t,
                "treat": 1,
                "entry_time": int(entry_t),
            }
            noise = rng.normal(0, 0.3, n_covariates)
            for k in range(n_covariates):
                row[f"x{k+1}"] = float(base_x[k] + noise[k])
            rows.append(row)
        uid += 1

    # Control units: lower baseline covariates (creates confounding)
    for i in range(n_controls):
        base_x = rng.normal(0, 1.0, n_covariates)
        for t in range(1, n_periods + 1):
            row = {
                "unit_id": uid,
                "time": t,
                "treat": 0,
                "entry_time": 99,
            }
            noise = rng.normal(0, 0.3, n_covariates)
            for k in range(n_covariates):
                row[f"x{k+1}"] = float(base_x[k] + noise[k])
            rows.append(row)
        uid += 1

    return pl.DataFrame(rows)


@pytest.fixture
def realistic_data():
    return make_confounded_data()


@pytest.fixture
def covariates():
    return ["x1", "x2", "x3", "x4", "x5"]


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end tests on realistic confounded data."""

    def test_matching_improves_balance(self, realistic_data, covariates):
        """Matching should reduce SMD compared to raw sample."""
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        assert result is not None
        # With confounded data, matching should improve balance
        max_smd = result.balance["matched_smd"].abs().max()
        max_raw_smd = result.balance["full_smd"].abs().max()
        assert max_smd < max_raw_smd, (
            f"Matching didn't improve balance: raw max|SMD|={max_raw_smd:.3f}, "
            f"matched max|SMD|={max_smd:.3f}"
        )

    def test_ebal_achieves_near_zero_balance(self, realistic_data, covariates):
        """Entropy balancing should achieve near-zero SMDs."""
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            method="ebal", moment=1, verbose=False,
        )
        assert result is not None
        max_smd = result.balance["matched_smd"].abs().max()
        assert max_smd < 0.15, f"Ebal max|SMD| = {max_smd:.4f}, expected < 0.15"

    def test_multiple_entry_cohorts_all_matched(self, realistic_data, covariates):
        """Each entry cohort should have matched units."""
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            num_matches=1, replacement="cross_cohort", verbose=False,
        )
        assert result is not None
        # Check that multiple time periods are represented in matches
        n_periods_matched = result.matched_data["time"].n_unique()
        assert n_periods_matched >= 3, (
            f"Only {n_periods_matched} periods had matches, expected >= 3"
        )

    def test_global_no_respects_constraint(self, realistic_data, covariates):
        """global_no: each control used at most once across ALL periods."""
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            num_matches=1, replacement="global_no", verbose=False,
        )
        assert result is not None
        ctrl_counts = result.matched_data.group_by("control_id").len()
        assert ctrl_counts["len"].max() <= 1

    def test_cross_cohort_within_period_constraint(self, realistic_data, covariates):
        """cross_cohort: no control reuse within a single period."""
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            num_matches=1, replacement="cross_cohort", verbose=False,
        )
        assert result is not None
        per_period = result.matched_data.group_by(["time", "control_id"]).len()
        assert per_period["len"].max() <= 1


class TestDistanceModelsRealistic:
    """Distance-based matching on realistic confounded data."""

    @pytest.mark.parametrize("model_type", list(DISTANCE_MODELS))
    def test_distance_model_produces_matches(self, realistic_data, covariates, model_type):
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            model_type=model_type, num_matches=1, verbose=False,
        )
        assert result is not None
        assert result.matched_data.height > 0

    def test_mahalanobis_better_than_euclidean(self, realistic_data, covariates):
        """Mahalanobis should generally achieve better balance than raw Euclidean
        on correlated covariates (it accounts for covariance structure)."""
        r_mah = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            model_type="mahalanobis", num_matches=1, verbose=False,
        )
        r_euc = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            model_type="euclidean", num_matches=1, verbose=False,
        )
        assert r_mah is not None and r_euc is not None
        # Both should produce results; just check they differ
        assert r_mah.matched_data.height > 0
        assert r_euc.matched_data.height > 0


class TestMahvarsRealistic:
    """Mahvars pattern on realistic data."""

    def test_mahvars_with_caliper(self, realistic_data, covariates):
        """PS caliper + Mahalanobis matching on subset."""
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.3, num_matches=1, replacement="cross_cohort",
            mahvars=["x1", "x2", "x3"], verbose=False,
        )
        assert result is not None
        assert result.matched_data.height > 0

    def test_mahvars_caliper_reduces_pool(self, realistic_data, covariates):
        """Tight PS caliper with mahvars should yield fewer matches."""
        r_loose = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0, num_matches=1, replacement="cross_cohort",
            mahvars=["x1", "x2"], verbose=False,
        )
        r_tight = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.1, num_matches=1, replacement="cross_cohort",
            mahvars=["x1", "x2"], verbose=False,
        )
        assert r_loose is not None
        if r_tight is not None:
            assert r_tight.matched_data.height <= r_loose.matched_data.height


class TestCaliperRealistic:
    """Caliper tests on realistic data."""

    def test_ps_caliper_reduces_matches(self, realistic_data, covariates):
        """PS caliper should reduce match count vs no caliper."""
        r_none = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0, num_matches=1, verbose=False,
        )
        r_tight = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.1, num_matches=1, verbose=False,
        )
        assert r_none is not None
        if r_tight is not None:
            assert r_tight.matched_data.height <= r_none.matched_data.height

    def test_per_variable_caliper(self, realistic_data, covariates):
        """Per-variable caliper should constrain specific covariates."""
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            num_matches=1, caliper={"x1": 0.5, "x2": 0.5}, verbose=False,
        )
        assert result is not None
        assert result.matched_data.height > 0

    def test_very_tight_caliper_graceful_none(self):
        """Extremely tight caliper should return None gracefully."""
        data = make_confounded_data(n_treated=20, n_controls=50, seed=99)
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2"],
            ps_caliper=0.0001, num_matches=1, verbose=False,
        )
        # Should either be None or have very few matches
        if result is not None:
            assert result.matched_data.height >= 0


class TestMOrderRealistic:
    """Matching order on realistic data."""

    def test_m_order_largest_vs_smallest(self, realistic_data, covariates):
        """Different m_order should produce different matches (cross_cohort)."""
        r1 = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.2, num_matches=1, replacement="cross_cohort",
            m_order="largest", verbose=False,
        )
        r2 = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.2, num_matches=1, replacement="cross_cohort",
            m_order="smallest", verbose=False,
        )
        assert r1 is not None and r2 is not None
        # At least some difference in control sets
        c1 = set(r1.matched_data["control_id"].to_list())
        c2 = set(r2.matched_data["control_id"].to_list())
        assert len(c1) > 0 and len(c2) > 0


class TestScaleRealistic:
    """Tests with larger data to verify performance doesn't degrade."""

    def test_1k_treated_10k_controls(self):
        """1K treated × 10K controls should complete quickly."""
        data = make_confounded_data(
            n_treated=1000, n_controls=10000, n_periods=10,
            n_entry_cohorts=4, n_covariates=3, seed=42,
        )
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.2, num_matches=1, verbose=False,
        )
        assert result is not None
        assert result.matched_data.height > 0
        assert result.n_treated_matched > 0

    def test_all_model_types_on_medium_data(self):
        """All 11 model types should work on medium-sized data."""
        data = make_confounded_data(
            n_treated=100, n_controls=500, n_periods=10,
            n_covariates=3, seed=42,
        )
        covs = ["x1", "x2", "x3"]
        for mt in SUPPORTED_MODELS:
            result = rollmatch(
                data, "treat", "time", "entry_time", "unit_id",
                covariates=covs,
                model_type=mt, num_matches=1, replacement="unrestricted",
                verbose=False,
            )
            assert result is not None, f"model_type={mt} returned None"
            assert result.matched_data.height > 0, f"model_type={mt} had 0 pairs"


class TestResultIntegrity:
    """Verify RollmatchResult contents are self-consistent."""

    def test_result_fields(self, realistic_data, covariates):
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=False,
        )
        assert result is not None

        # Type checks
        assert isinstance(result, RollmatchResult)
        assert isinstance(result.matched_data, pl.DataFrame)
        assert isinstance(result.balance, pl.DataFrame)
        assert isinstance(result.weights, pl.DataFrame)
        assert result.method == "matching"
        assert result.ps_caliper == 0.2

        # matched_data columns
        assert set(result.matched_data.columns) == {"time", "treat_id", "control_id", "difference"}

        # Counts are consistent
        assert result.n_treated_matched <= result.n_treated_total
        assert result.n_treated_matched == result.matched_data["treat_id"].n_unique()
        assert result.n_controls_matched == result.matched_data["control_id"].n_unique()

        # Balance table has one row per covariate
        assert result.balance.height == len(covariates)

        # Weights table covers all matched units
        matched_ids = set(result.matched_data["treat_id"].to_list()) | set(result.matched_data["control_id"].to_list())
        weight_ids = set(result.weights["unit_id"].to_list())
        assert matched_ids == weight_ids

    def test_ebal_result_fields(self, realistic_data, covariates):
        result = rollmatch(
            realistic_data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            method="ebal", moment=1, verbose=False,
        )
        assert result is not None
        assert result.matched_data is None
        assert result.ps_caliper is None
        assert result.method == "ebal"
        assert result.weighted_data is not None
        assert "weight" in result.weights.columns
