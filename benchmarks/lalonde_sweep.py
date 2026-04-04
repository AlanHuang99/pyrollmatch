"""Sweep Lalonde-based matching configurations and report balance.

Creates pseudo rolling-entry panels from MatchIt's Lalonde data and evaluates:
- all supported matching models under common replacement modes
- a small set of targeted logistic tuning specs
- entropy balancing moments 1 and 2

Outputs:
- benchmarks/lalonde_sweep_results.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import polars as pl
from sklearn.exceptions import ConvergenceWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyrollmatch import (
    SUPPORTED_MODELS,
    balance_by_period,
    balance_by_period_weighted,
    reduce_data,
    rollmatch,
    score_data,
)
from tests.real_world import REAL_WORLD_COVARIATES, make_lalonde_panel


OUTPUT_PATH = Path(__file__).with_name("lalonde_sweep_results.csv")
MATCHING_MODELS = {
    "logistic", "probit", "gbm", "rf", "lasso", "ridge", "elasticnet",
}


def _evaluate_matching(
    data: pl.DataFrame,
    cohort_strategy: str,
    name: str,
    **kwargs,
) -> dict:
    """Run one matching spec and compute pooled/period balance metrics."""
    model_type = kwargs.get("model_type", "logistic")
    reduced = reduce_data(
        data, "treat", "time", "entry_time", "unit_id",
    ).drop_nulls(subset=REAL_WORLD_COVARIATES)
    scored = score_data(
        reduced, REAL_WORLD_COVARIATES, "treat", model_type=model_type,
    ).data
    result = rollmatch(
        data, "treat", "time", "entry_time", "unit_id",
        covariates=REAL_WORLD_COVARIATES,
        **kwargs,
    )
    if result is None:
        return {
            "cohort_strategy": cohort_strategy,
            "kind": "matching",
            "name": name,
            "model_type": model_type,
            "replacement": kwargs.get("replacement"),
            "num_matches": kwargs.get("num_matches", 1),
            "ps_caliper": kwargs.get("ps_caliper"),
            "method": "matching",
            "n_treated_matched": 0,
            "n_treated_total": reduced.filter(pl.col("treat") == 1)["unit_id"].n_unique(),
            "match_rate": 0.0,
            "pooled_max_smd": None,
            "period_max_smd": None,
            "pooled_pass_0_10": False,
            "period_pass_0_10": False,
            "worst_period_covariate": None,
        }
    agg, _ = balance_by_period(
        scored, result.matched_data, "treat", "unit_id", "time", REAL_WORLD_COVARIATES,
    )
    return {
        "cohort_strategy": cohort_strategy,
        "kind": "matching",
        "name": name,
        "model_type": model_type,
        "replacement": kwargs.get("replacement"),
        "num_matches": kwargs.get("num_matches", 1),
        "ps_caliper": kwargs.get("ps_caliper"),
        "method": "matching",
        "n_treated_matched": result.n_treated_matched,
        "n_treated_total": result.n_treated_total,
        "match_rate": round(result.n_treated_matched / result.n_treated_total, 4),
        "pooled_max_smd": round(float(result.balance["matched_smd"].abs().max()), 4),
        "period_max_smd": round(float(agg["max_abs_smd"].max()), 4),
        "pooled_pass_0_10": bool(result.balance["matched_smd"].abs().max() < 0.10),
        "period_pass_0_10": bool(agg["max_abs_smd"].max() < 0.10),
        "worst_period_covariate": agg.sort("max_abs_smd", descending=True)["covariate"][0],
    }


def _evaluate_ebal(
    data: pl.DataFrame,
    cohort_strategy: str,
    moment: int,
) -> dict:
    """Run one entropy balancing spec and compute pooled/period balance."""
    reduced = reduce_data(
        data, "treat", "time", "entry_time", "unit_id",
    ).drop_nulls(subset=REAL_WORLD_COVARIATES)
    result = rollmatch(
        data, "treat", "time", "entry_time", "unit_id",
        covariates=REAL_WORLD_COVARIATES,
        method="ebal", moment=moment, verbose=False,
    )
    if result is None:
        return {
            "cohort_strategy": cohort_strategy,
            "kind": "ebal",
            "name": f"ebal_m{moment}",
            "model_type": None,
            "replacement": None,
            "num_matches": None,
            "ps_caliper": None,
            "method": "ebal",
            "n_treated_matched": 0,
            "n_treated_total": reduced.filter(pl.col("treat") == 1)["unit_id"].n_unique(),
            "match_rate": 0.0,
            "pooled_max_smd": None,
            "period_max_smd": None,
            "pooled_pass_0_10": False,
            "period_pass_0_10": False,
            "worst_period_covariate": None,
        }
    agg, _ = balance_by_period_weighted(
        reduced, result.weighted_data, "treat", "unit_id", "time", REAL_WORLD_COVARIATES,
    )
    return {
        "cohort_strategy": cohort_strategy,
        "kind": "ebal",
        "name": f"ebal_m{moment}",
        "model_type": None,
        "replacement": None,
        "num_matches": None,
        "ps_caliper": None,
        "method": "ebal",
        "n_treated_matched": result.n_treated_matched,
        "n_treated_total": result.n_treated_total,
        "match_rate": round(result.n_treated_matched / result.n_treated_total, 4),
        "pooled_max_smd": round(float(result.balance["matched_smd"].abs().max()), 4),
        "period_max_smd": round(float(agg["max_abs_smd"].max()), 4),
        "pooled_pass_0_10": bool(result.balance["matched_smd"].abs().max() < 0.10),
        "period_pass_0_10": bool(agg["max_abs_smd"].max() < 0.10),
        "worst_period_covariate": agg.sort("max_abs_smd", descending=True)["covariate"][0],
    }


def run_sweep() -> pl.DataFrame:
    """Run the full Lalonde validation sweep."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

    targeted_specs = {
        "logistic_ps010_1to1": dict(
            model_type="logistic", ps_caliper=0.10,
            replacement="cross_cohort", num_matches=1, verbose=False,
        ),
        "logistic_ps020_1to1": dict(
            model_type="logistic", ps_caliper=0.20,
            replacement="cross_cohort", num_matches=1, verbose=False,
        ),
        "logistic_ps030_1to1": dict(
            model_type="logistic", ps_caliper=0.30,
            replacement="cross_cohort", num_matches=1, verbose=False,
        ),
        "logistic_ps020_2to1": dict(
            model_type="logistic", ps_caliper=0.20,
            replacement="cross_cohort", num_matches=2, verbose=False,
        ),
        "logistic_var_caliper": dict(
            model_type="logistic", ps_caliper=0.20,
            replacement="cross_cohort", num_matches=1,
            caliper={"age": 0.5, "educ": 0.5}, verbose=False,
        ),
        "logistic_mahvars": dict(
            model_type="logistic", ps_caliper=0.20,
            replacement="cross_cohort", num_matches=1,
            mahvars=["age", "educ", "re74_k", "re75_k"], verbose=False,
        ),
    }

    rows = []
    for cohort_strategy in ("age_split", "hash_split"):
        data = make_lalonde_panel(cohort_strategy=cohort_strategy)

        for model_type in SUPPORTED_MODELS:
            for replacement in ("unrestricted", "cross_cohort", "global_no"):
                kwargs = dict(
                    model_type=model_type,
                    replacement=replacement,
                    num_matches=1,
                    verbose=False,
                )
                kwargs["ps_caliper"] = 0.20 if model_type in MATCHING_MODELS else 0.0
                rows.append(
                    _evaluate_matching(
                        data, cohort_strategy, f"{model_type}_{replacement}", **kwargs,
                    )
                )

        for name, kwargs in targeted_specs.items():
            rows.append(_evaluate_matching(data, cohort_strategy, name, **kwargs))

        for moment in (1, 2):
            rows.append(_evaluate_ebal(data, cohort_strategy, moment))

    return pl.DataFrame(rows)


def main() -> None:
    """Run sweep, print a concise summary, and persist CSV results."""
    results = run_sweep()
    results.write_csv(OUTPUT_PATH)

    print(f"Saved results to {OUTPUT_PATH}")

    for cohort_strategy in ("age_split", "hash_split"):
        cohort_results = results.filter(pl.col("cohort_strategy") == cohort_strategy)
        print(f"\n=== {cohort_strategy} ===")
        print("\nTop by pooled balance:")
        print(
            cohort_results
            .sort(["pooled_max_smd", "match_rate"], descending=[False, True])
            .select([
                "kind", "name", "match_rate", "pooled_max_smd",
                "period_max_smd", "pooled_pass_0_10", "period_pass_0_10",
            ])
            .head(10)
        )
        print("\nTop by period balance:")
        print(
            cohort_results
            .sort(["period_max_smd", "pooled_max_smd"], descending=[False, True])
            .select([
                "kind", "name", "match_rate", "pooled_max_smd",
                "period_max_smd", "pooled_pass_0_10", "period_pass_0_10",
            ])
            .head(10)
        )
        print("\nPass counts:")
        print(
            cohort_results.group_by("kind").agg([
                pl.len().alias("n_specs"),
                pl.col("pooled_pass_0_10").sum().alias("pooled_lt_0_10"),
                pl.col("period_pass_0_10").sum().alias("period_lt_0_10"),
            ])
        )


if __name__ == "__main__":
    main()
