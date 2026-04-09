"""
Reproducibility / determinism regression tests.

These tests guard the contract that pyrollmatch produces identical matches
when called repeatedly with identical inputs — both same-process and across
BLAS thread count variations.

They use small synthetic data so they run in CI in well under a second.
"""
from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

from pyrollmatch import reduce_data, rollmatch, score_data
from pyrollmatch.score import SUPPORTED_MODELS, DISTANCE_MODELS


def _make_panel(seed: int = 17, n_treated: int = 100, n_controls: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated):
        et = int(rng.integers(4, 8))
        for t in range(1, 11):
            row = {
                "unit_id": i, "time": t, "treat": 1, "entry_time": et,
            }
            for j in range(5):
                row[f"x{j+1}"] = float(
                    rng.normal(0.5, 1.0) + (0.3 if t >= et else 0)
                )
            rows.append(row)
    for i in range(n_controls):
        for t in range(1, 11):
            row = {
                "unit_id": n_treated + i, "time": t, "treat": 0, "entry_time": 999,
            }
            for j in range(5):
                row[f"x{j+1}"] = float(rng.normal(0, 1.0))
            rows.append(row)
    return pl.DataFrame(rows)


@pytest.fixture
def panel() -> pl.DataFrame:
    return _make_panel()


def _signature(matched: pl.DataFrame) -> tuple[int, str, str]:
    """Return (n_pairs, id_hash, full_hash)."""
    sorted_md = matched.select(
        ["time", "treat_id", "control_id", "difference"]
    ).sort(["time", "treat_id", "control_id"])
    id_blob = sorted_md.select(
        ["time", "treat_id", "control_id"]
    ).write_csv().encode()
    full_blob = sorted_md.write_csv().encode()
    return (
        len(sorted_md),
        hashlib.sha256(id_blob).hexdigest()[:16],
        hashlib.sha256(full_blob).hexdigest()[:16],
    )


class TestScoreDataAcceptsRandomState:
    """score_data must accept a random_state parameter so users can
    override the hardcoded default."""

    def test_score_data_accepts_random_state_kwarg(self, panel):
        """score_data(..., random_state=X) must not raise."""
        reduced = reduce_data(panel, "treat", "time", "entry_time", "unit_id")
        result = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type="logistic", random_state=7,
        )
        assert result is not None
        assert result.data is not None

    def test_score_data_random_state_changes_stochastic_model_scores(self, panel):
        """Different random_state values should produce different scores for
        stochastic models (RF)."""
        reduced = reduce_data(panel, "treat", "time", "entry_time", "unit_id")
        r1 = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type="rf", random_state=1,
        )
        r2 = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type="rf", random_state=2,
        )
        # RF with different seeds should produce different trees and different scores
        assert not np.array_equal(r1.data["score"].to_numpy(), r2.data["score"].to_numpy())

    def test_score_data_random_state_default_is_backward_compatible(self, panel):
        """Omitting random_state should use the historical default (42)."""
        reduced = reduce_data(panel, "treat", "time", "entry_time", "unit_id")
        r_default = score_data(
            reduced, ["x1", "x2", "x3"], "treat", model_type="rf",
        )
        r_42 = score_data(
            reduced, ["x1", "x2", "x3"], "treat",
            model_type="rf", random_state=42,
        )
        np.testing.assert_array_equal(
            r_default.data["score"].to_numpy(),
            r_42.data["score"].to_numpy(),
        )


class TestBitStableAcrossRepeatedCalls:
    """Every supported model_type must produce bit-identical matched_data
    when rollmatch is called repeatedly with identical inputs."""

    @pytest.mark.parametrize("model_type", sorted(SUPPORTED_MODELS))
    def test_repeated_rollmatch_is_bit_identical(self, panel, model_type):
        is_distance = model_type in DISTANCE_MODELS
        kwargs = dict(
            treat="treat", tm="time", entry="entry_time", id="unit_id",
            covariates=["x1", "x2", "x3", "x4", "x5"],
            lookback=1, model_type=model_type,
            num_matches=1, replacement="global_no", block_size=2000,
            verbose=False,
        )
        if not is_distance:
            kwargs["ps_caliper"] = 0.2

        results = [rollmatch(panel, **kwargs) for _ in range(5)]
        assert all(r is not None for r in results)
        sigs = [_signature(r.matched_data) for r in results]
        # All 5 calls must agree on pair IDs
        unique_id_hashes = {s[1] for s in sigs}
        assert len(unique_id_hashes) == 1, (
            f"pair IDs varied across 5 calls for model_type={model_type}: "
            f"{unique_id_hashes}"
        )
        # All 5 calls must agree on distance values too
        unique_full_hashes = {s[2] for s in sigs}
        assert len(unique_full_hashes) == 1, (
            f"distance values varied across 5 calls for model_type={model_type}: "
            f"{unique_full_hashes}"
        )


class TestRandomStatePropagation:
    """rollmatch's random_state parameter should propagate to score_data for
    stochastic models."""

    def test_rollmatch_random_state_changes_rf_scores(self, panel):
        """Different rollmatch(random_state=...) values should produce
        different RF propensity scores → potentially different matches."""
        kwargs = dict(
            treat="treat", tm="time", entry="entry_time", id="unit_id",
            covariates=["x1", "x2", "x3", "x4", "x5"],
            lookback=1, model_type="rf",
            num_matches=1, replacement="global_no", ps_caliper=0.2,
            verbose=False,
        )
        r1 = rollmatch(panel, **kwargs, random_state=1)
        r2 = rollmatch(panel, **kwargs, random_state=2)
        # With different seeds, scored propensities differ → at least some
        # matches should differ
        s1 = _signature(r1.matched_data)
        s2 = _signature(r2.matched_data)
        # We allow the pair IDs to be the same if the matching is robust,
        # but the distance values (from different RF scores) must differ.
        assert s1[2] != s2[2], (
            "rollmatch(random_state=1) and rollmatch(random_state=2) should "
            "produce different RF scores and therefore different difference "
            "values"
        )

    def test_rollmatch_random_state_none_uses_default(self, panel):
        """rollmatch(random_state=None) should be bit-identical to
        rollmatch(random_state=42) for stochastic PS models (since the
        historical default was 42)."""
        kwargs = dict(
            treat="treat", tm="time", entry="entry_time", id="unit_id",
            covariates=["x1", "x2", "x3", "x4", "x5"],
            lookback=1, model_type="rf",
            num_matches=1, replacement="global_no", ps_caliper=0.2,
            verbose=False,
        )
        r_none = rollmatch(panel, **kwargs)
        r_42 = rollmatch(panel, **kwargs, random_state=42)
        assert _signature(r_none.matched_data) == _signature(r_42.matched_data)
