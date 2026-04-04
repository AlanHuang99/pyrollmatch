"""Stress tests — large-scale performance verification."""

import polars as pl
import numpy as np
import pytest
import time
from pyrollmatch import rollmatch
from tests.test_smoke import make_synthetic_data


class TestLargeScale:
    @pytest.mark.parametrize("n_treated,n_controls", [
        (1000, 3000),
        (5000, 15000),
    ])
    def test_scales(self, n_treated, n_controls):
        """Test that matching scales to thousands of units."""
        data = make_synthetic_data(n_treated, n_controls, n_periods=15, seed=42)

        start = time.time()
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.1, num_matches=3, replacement="unrestricted",
            verbose=True,
        )
        elapsed = time.time() - start

        assert result is not None
        print(f"\n  {n_treated}×{n_controls}: {elapsed:.1f}s, "
              f"matched={result.n_treated_matched}/{result.n_treated_total}")

    def test_10k_treated(self):
        """Test with 10K treated — our target scale."""
        data = make_synthetic_data(10000, 30000, n_periods=15, seed=42)

        start = time.time()
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            ps_caliper=0.1, num_matches=3, replacement="unrestricted",
            block_size=2000, verbose=True,
        )
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 300  # Should complete in under 5 minutes
        print(f"\n  10K×30K: {elapsed:.1f}s")

    def test_many_covariates(self):
        """Test with many covariates."""
        rng = np.random.default_rng(42)
        n_treated, n_controls, n_periods = 500, 1500, 10
        n_covs = 20

        rows = []
        for i in range(n_treated + n_controls):
            is_treat = i < n_treated
            entry_t = rng.integers(6, 10) if is_treat else 99
            for t in range(1, n_periods + 1):
                row = {
                    "unit_id": i, "time": t,
                    "treat": 1 if is_treat else 0,
                    "entry_time": int(entry_t),
                }
                for c in range(n_covs):
                    row[f"x{c}"] = float(rng.exponential(1.0))
                rows.append(row)

        data = pl.DataFrame(rows)
        covariates = [f"x{c}" for c in range(n_covs)]

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates,
            ps_caliper=0.2, num_matches=3, replacement="unrestricted",
            verbose=True,
        )
        assert result is not None
        assert result.balance.height == n_covs
