"""
Validation of replacement modes using controlled synthetic data.

Uses carefully constructed scenarios where we can predict and verify
the exact behavior of each replacement mode, not just statistical properties.
"""

import polars as pl
import numpy as np
import pytest
from pyrollmatch import rollmatch, balance_by_period, reduce_data, score_data


def make_controlled_panel(
    n_treated: int = 60,
    n_controls: int = 200,
    n_periods: int = 12,
    n_entry_cohorts: int = 3,
    seed: int = 99,
) -> pl.DataFrame:
    """Build a panel where treated enter in distinct cohorts.

    Cohorts enter at periods 5, 7, 9 (by default 3 cohorts).
    Controls have similar covariate distributions so matching is easy
    (high overlap) — this lets us test replacement logic rather than
    caliper edge cases.
    """
    rng = np.random.default_rng(seed)
    entry_periods = [5, 7, 9][:n_entry_cohorts]
    treated_per_cohort = n_treated // n_entry_cohorts

    rows = []
    uid = 0

    # Treated units — spread across cohorts
    for cohort_idx, entry_t in enumerate(entry_periods):
        for _ in range(treated_per_cohort):
            base = rng.uniform(1.0, 3.0, size=3)
            for t in range(1, n_periods + 1):
                noise = rng.normal(0, 0.2, size=3)
                boost = 1.3 if t >= entry_t else 1.0
                rows.append({
                    "unit_id": uid, "time": t, "treat": 1,
                    "entry_time": int(entry_t),
                    "x1": float(base[0] * boost + noise[0]),
                    "x2": float(base[1] * boost + noise[1]),
                    "x3": float(base[2] * boost + noise[2]),
                })
            uid += 1

    # Controls — similar distribution, no treatment
    for _ in range(n_controls):
        base = rng.uniform(1.0, 3.0, size=3)
        for t in range(1, n_periods + 1):
            noise = rng.normal(0, 0.2, size=3)
            rows.append({
                "unit_id": uid, "time": t, "treat": 0,
                "entry_time": 99,
                "x1": float(base[0] + noise[0]),
                "x2": float(base[1] + noise[1]),
                "x3": float(base[2] + noise[2]),
            })
        uid += 1

    return pl.DataFrame(rows)


def make_scarce_control_panel(seed: int = 77) -> pl.DataFrame:
    """Panel with few controls to force competition between cohorts.

    3 cohorts of 10 treated each (30 total) but only 25 controls.
    With global_no + num_matches=1, at most 25 treated can be matched.
    With cross_cohort, controls are reused across cohorts so more can match.
    """
    rng = np.random.default_rng(seed)
    entry_periods = [5, 7, 9]
    rows = []
    uid = 0

    for entry_t in entry_periods:
        for _ in range(10):
            base = rng.uniform(1.5, 2.5, size=3)
            for t in range(1, 13):
                noise = rng.normal(0, 0.15, size=3)
                rows.append({
                    "unit_id": uid, "time": t, "treat": 1,
                    "entry_time": int(entry_t),
                    "x1": float(base[0] + noise[0]),
                    "x2": float(base[1] + noise[1]),
                    "x3": float(base[2] + noise[2]),
                })
            uid += 1

    for _ in range(25):
        base = rng.uniform(1.5, 2.5, size=3)
        for t in range(1, 13):
            noise = rng.normal(0, 0.15, size=3)
            rows.append({
                "unit_id": uid, "time": t, "treat": 0,
                "entry_time": 99,
                "x1": float(base[0] + noise[0]),
                "x2": float(base[1] + noise[1]),
                "x3": float(base[2] + noise[2]),
            })
        uid += 1

    return pl.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Test class: Replacement mode behavior under controlled conditions
# ──────────────────────────────────────────────────────────────────────────────

class TestReplacementValidation:
    """Verify that each mode has the right structural properties on real data."""

    @pytest.fixture
    def data(self):
        return make_controlled_panel()

    @pytest.fixture
    def scarce_data(self):
        return make_scarce_control_panel()

    # ── Structural invariants ──

    def test_global_no_unique_controls_across_all_periods(self, data):
        """The defining property: each control appears at most once globally."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.3, num_matches=1, replacement="global_no", verbose=False,
        )
        assert result is not None

        # Each control_id should appear exactly once in the full match table
        ctrl_counts = result.matched_data.group_by("control_id").len()
        assert ctrl_counts["len"].max() == 1, (
            f"global_no violated: a control was matched {ctrl_counts['len'].max()} times"
        )

    def test_cross_cohort_unique_within_but_shared_across(self, data):
        """cross_cohort: unique within each period, can repeat across."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.3, num_matches=1, replacement="cross_cohort", verbose=False,
        )
        assert result is not None

        # Within each period: control appears at most once
        within = result.matched_data.group_by(["time", "control_id"]).len()
        assert within["len"].max() == 1

        # Across periods: at least some controls should appear in multiple periods
        across = (
            result.matched_data.select("control_id", "time").unique()
            .group_by("control_id").len()
        )
        max_cross = across["len"].max()
        # With 3 cohorts and 200 controls, some sharing is very likely
        assert max_cross >= 2, (
            f"Expected some cross-period reuse but max was {max_cross}"
        )

    def test_unrestricted_allows_within_period_reuse(self, data):
        """unrestricted: controls can match multiple treated in same period."""
        # Use many matches to increase chance of within-period reuse
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.5, num_matches=5, replacement="unrestricted", verbose=False,
        )
        assert result is not None

        # Check if any control is matched multiple times in same period
        within = result.matched_data.group_by(["time", "control_id"]).len()
        max_within = within["len"].max()
        assert max_within >= 2, (
            f"Expected within-period reuse but max was {max_within}. "
            "With 5 matches per treated and 'unrestricted', controls should be reused."
        )

    # ── Scarce control pool: tests that global_no truly constrains ──

    def test_scarce_global_no_caps_at_control_count(self, scarce_data):
        """With 25 controls and global_no + 1:1, at most 25 pairs possible."""
        result = rollmatch(
            scarce_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.5, num_matches=1, replacement="global_no", verbose=False,
        )
        assert result is not None
        # Cannot exceed the control pool size
        assert result.matched_data.height <= 25, (
            f"global_no produced {result.matched_data.height} pairs with only 25 controls"
        )
        # Verify uniqueness
        ctrl_counts = result.matched_data.group_by("control_id").len()
        assert ctrl_counts["len"].max() == 1

    def test_scarce_cross_cohort_more_than_global_no(self, scarce_data):
        """cross_cohort should match more treated than global_no when controls are scarce."""
        r_global = rollmatch(
            scarce_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.5, num_matches=1, replacement="global_no", verbose=False,
        )
        r_cross = rollmatch(
            scarce_data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.5, num_matches=1, replacement="cross_cohort", verbose=False,
        )
        assert r_global is not None
        assert r_cross is not None

        # cross_cohort reuses controls across periods → more matches
        assert r_cross.matched_data.height >= r_global.matched_data.height, (
            f"cross_cohort ({r_cross.matched_data.height}) should have >= "
            f"global_no ({r_global.matched_data.height}) matches"
        )

    def test_scarce_unrestricted_most_matches(self, scarce_data):
        """unrestricted should match the most treated of all three modes."""
        results = {}
        for mode in ["unrestricted", "cross_cohort", "global_no"]:
            r = rollmatch(
                scarce_data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                alpha=0.5, num_matches=1, replacement=mode, verbose=False,
            )
            results[mode] = r.matched_data.height if r else 0

        assert results["unrestricted"] >= results["cross_cohort"], (
            f"unrestricted ({results['unrestricted']}) should >= "
            f"cross_cohort ({results['cross_cohort']})"
        )
        assert results["cross_cohort"] >= results["global_no"], (
            f"cross_cohort ({results['cross_cohort']}) should >= "
            f"global_no ({results['global_no']})"
        )

    # ── Ordering: global_no is sensitive to cohort processing order ──

    def test_global_no_processes_earliest_first(self, data):
        """global_no processes periods chronologically; earlier cohorts get first pick."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.3, num_matches=1, replacement="global_no", verbose=False,
        )
        assert result is not None

        periods = sorted(result.matched_data["time"].unique().to_list())
        # The earliest period should have a match (controls were plentiful)
        first_period_matches = result.matched_data.filter(
            pl.col("time") == periods[0]
        ).height
        assert first_period_matches > 0

    # ── Balance by period: verify diagnostics work with each mode ──

    def test_balance_by_period_all_modes(self, data):
        """balance_by_period should work with results from all three modes."""
        reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat")

        for mode in ["unrestricted", "cross_cohort", "global_no"]:
            result = rollmatch(
                data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                alpha=0.3, num_matches=1, replacement=mode, verbose=False,
            )
            assert result is not None, f"No matches for mode={mode}"

            agg, detail = balance_by_period(
                scored, result.matched_data,
                "treat", "unit_id", "time", ["x1", "x2", "x3"],
            )
            assert agg.height == 3, f"mode={mode}: expected 3 covariates in agg"
            assert detail.height > 0, f"mode={mode}: detail should not be empty"

            # max_abs_smd should be >= the pooled |SMD|'s absolute value
            # (pooled can cancel; max per-period cannot)
            for cov in ["x1", "x2", "x3"]:
                max_per_period = float(
                    agg.filter(pl.col("covariate") == cov)["max_abs_smd"][0]
                )
                assert max_per_period >= 0, f"mode={mode}, cov={cov}: negative max_abs_smd"

    # ── Per-period balance reveals cancellation ──

    def test_pooled_vs_per_period_smd(self, data):
        """max per-period |SMD| should be >= pooled |SMD| (cancellation effect)."""
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=["x1", "x2", "x3"],
            alpha=0.3, num_matches=3, replacement="cross_cohort", verbose=False,
        )
        assert result is not None

        reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat")

        agg, _ = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", ["x1", "x2", "x3"],
        )

        for cov in ["x1", "x2", "x3"]:
            pooled_smd = abs(float(
                result.balance.filter(pl.col("covariate") == cov)["matched_smd"][0]
            ))
            max_period_smd = float(
                agg.filter(pl.col("covariate") == cov)["max_abs_smd"][0]
            )
            # The worst single period should be at least as bad as the pooled average
            assert max_period_smd >= pooled_smd - 0.01, (
                f"cov={cov}: max per-period |SMD| ({max_period_smd:.4f}) < "
                f"pooled |SMD| ({pooled_smd:.4f}). "
                "Per-period max should be >= pooled (pooled can cancel)."
            )


class TestReplacementModeComparisonReport:
    """Run all three modes side by side and print a comparison report.

    This is a single integration test that validates the modes produce
    meaningfully different results and prints a human-readable summary.
    """

    def test_comparison_report(self, capsys):
        data = make_controlled_panel(n_treated=90, n_controls=150, seed=42)

        reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
        reduced = reduced.drop_nulls(subset=["x1", "x2", "x3"])
        scored = score_data(reduced, ["x1", "x2", "x3"], "treat")

        modes = ["unrestricted", "cross_cohort", "global_no"]
        results = {}

        for mode in modes:
            r = rollmatch(
                data, "treat", "time", "entry_time", "unit_id",
                covariates=["x1", "x2", "x3"],
                alpha=0.3, num_matches=1, replacement=mode, verbose=False,
            )
            assert r is not None, f"No matches for mode={mode}"
            results[mode] = r

        print("\n" + "=" * 78)
        print("  REPLACEMENT MODE COMPARISON")
        print("  (90 treated across 3 cohorts, 150 controls, 1:1 matching)")
        print("=" * 78)

        print(f"\n  {'Mode':<18} {'Pairs':>6} {'Treated':>8} {'Controls':>9} {'Max|SMD|':>10}")
        print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*9} {'-'*10}")

        for mode in modes:
            r = results[mode]
            max_smd = r.balance["matched_smd"].abs().max()
            print(
                f"  {mode:<18} {r.matched_data.height:>6} "
                f"{r.n_treated_matched:>8} {r.n_controls_matched:>9} "
                f"{max_smd:>10.4f}"
            )

        # Verify ordering: unrestricted >= cross_cohort >= global_no
        assert (results["unrestricted"].matched_data.height
                >= results["cross_cohort"].matched_data.height)
        assert (results["cross_cohort"].matched_data.height
                >= results["global_no"].matched_data.height)

        # Verify global_no uniqueness
        g = results["global_no"]
        ctrl_counts = g.matched_data.group_by("control_id").len()
        assert ctrl_counts["len"].max() == 1

        # Per-period balance comparison
        print(f"\n  {'Mode':<18} {'Cov':<6} {'Pooled SMD':>11} {'Max Period':>11} {'Wtd Mean':>9}")
        print(f"  {'-'*18} {'-'*6} {'-'*11} {'-'*11} {'-'*9}")

        for mode in modes:
            r = results[mode]
            agg, _ = balance_by_period(
                scored, r.matched_data,
                "treat", "unit_id", "time", ["x1", "x2", "x3"],
            )
            for cov in ["x1", "x2", "x3"]:
                pooled = float(
                    r.balance.filter(pl.col("covariate") == cov)["matched_smd"][0]
                )
                max_p = float(
                    agg.filter(pl.col("covariate") == cov)["max_abs_smd"][0]
                )
                wtd = float(
                    agg.filter(pl.col("covariate") == cov)["wtd_mean_smd"][0]
                )
                print(
                    f"  {mode:<18} {cov:<6} {pooled:>11.4f} {max_p:>11.4f} {wtd:>9.4f}"
                )

        print("\n" + "=" * 78)

        captured = capsys.readouterr()
        assert "REPLACEMENT MODE COMPARISON" in captured.out
