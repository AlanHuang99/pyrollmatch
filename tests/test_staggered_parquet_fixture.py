"""Tests for the large staggered parquet fixture."""

import polars as pl

from pyrollmatch import (
    balance_by_period,
    balance_by_period_weighted,
    reduce_data,
    rollmatch,
    score_data,
)
from tests.staggered_data import (
    STAGGERED_PANEL_LARGE_COVARIATES,
    load_staggered_panel_large,
    load_staggered_panel_large_metadata,
)


class TestStaggeredParquetFixture:
    def test_fixture_structure_and_metadata(self):
        data = load_staggered_panel_large()
        meta = load_staggered_panel_large_metadata()

        assert data.height == meta["n_rows"]
        assert data.width == len(meta["columns"])
        assert data.columns == meta["columns"]
        assert data["unit_id"].n_unique() == meta["n_units"]
        assert set(data["treat"].unique().to_list()) == {0, 1}

        per_unit = data.group_by("unit_id").len()
        assert per_unit["len"].min() == meta["n_periods"]
        assert per_unit["len"].max() == meta["n_periods"]

        treated = data.filter(pl.col("treat") == 1).select("unit_id", "entry_time").unique()
        controls = data.filter(pl.col("treat") == 0).select("unit_id", "entry_time").unique()

        assert treated.height == meta["n_treated_units"]
        assert controls.height == meta["n_control_units"]
        assert controls["entry_time"].unique().to_list() == [meta["control_entry_sentinel"]]
        assert treated["entry_time"].unique().sort().to_list() == meta["entry_periods"]

    def test_matching_on_large_parquet_fixture(self):
        data = load_staggered_panel_large()

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=STAGGERED_PANEL_LARGE_COVARIATES,
            model_type="logistic",
            ps_caliper=0.25,
            replacement="cross_cohort",
            num_matches=1,
            verbose=False,
        )

        assert result is not None
        assert result.n_treated_matched / result.n_treated_total > 0.99

        full_max = result.balance["full_smd"].abs().max()
        matched_max = result.balance["matched_smd"].abs().max()
        assert matched_max < full_max
        assert matched_max < 0.05

        reduced = reduce_data(
            data, "treat", "time", "entry_time", "unit_id",
        ).drop_nulls(subset=STAGGERED_PANEL_LARGE_COVARIATES)
        scored = score_data(
            reduced, STAGGERED_PANEL_LARGE_COVARIATES, "treat",
            model_type="logistic",
        ).data
        agg, _ = balance_by_period(
            scored, result.matched_data,
            "treat", "unit_id", "time", STAGGERED_PANEL_LARGE_COVARIATES,
        )
        assert agg["max_abs_smd"].max() < 0.20

    def test_ebal_on_large_parquet_fixture(self):
        data = load_staggered_panel_large()
        meta = load_staggered_panel_large_metadata()

        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=STAGGERED_PANEL_LARGE_COVARIATES,
            method="ebal",
            moment=1,
            verbose=False,
        )

        assert result is not None
        assert result.n_treated_matched == meta["n_treated_units"]
        assert result.balance["matched_smd"].abs().max() < 0.02

        reduced = reduce_data(
            data, "treat", "time", "entry_time", "unit_id",
        ).drop_nulls(subset=STAGGERED_PANEL_LARGE_COVARIATES)
        agg, _ = balance_by_period_weighted(
            reduced, result.weighted_data,
            "treat", "unit_id", "time", STAGGERED_PANEL_LARGE_COVARIATES,
        )
        assert agg["max_abs_smd"].max() < 0.01
