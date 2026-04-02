"""
core — Main rollmatch orchestration with pluggable method dispatch.

Supports matching (propensity score nearest-neighbor) and weighting
(entropy balancing) methods through a unified interface. New methods
can be added as callables conforming to the per-period interface.
"""

import warnings
from typing import Callable

import polars as pl
import numpy as np
from dataclasses import dataclass

from .reduce import reduce_data
from .score import score_data
from .match import match_all_periods
from .weight import _compute_weights, entropy_balance
from .balance import compute_balance, compute_balance_weighted, smd_table


@dataclass
class RollmatchResult:
    """Result from rollmatch.

    Attributes
    ----------
    matched_data : pl.DataFrame or None
        Match pairs [tm, treat_id, control_id, difference].
        None for weighting-only methods.
    balance : pl.DataFrame
        Covariate balance table (pooled or per-period depending on method).
    n_treated_total : int
        Total treated units in the reduced sample.
    n_treated_matched : int
        Treated units successfully matched/weighted.
    n_controls_matched : int
        Unique control units used.
    alpha : float or None
        Caliper multiplier. None for non-caliper methods.
    weights : pl.DataFrame
        Unit weights [id, weight]. For matching, derived from pairs.
        For ebal, collapsed across cohorts (convenience — use
        ``weighted_data`` for per-cohort weights).
    weighted_data : pl.DataFrame or None
        Stacked per-cohort weights [tm, id, weight]. Each cohort has
        its own weight vector; controls appear once per cohort they
        participate in. None for matching.
    method : str
        Method used: "matching", "ebal", "custom".
    """
    matched_data: pl.DataFrame | None
    balance: pl.DataFrame
    n_treated_total: int
    n_treated_matched: int
    n_controls_matched: int
    alpha: float | None
    weights: pl.DataFrame
    weighted_data: pl.DataFrame | None = None
    method: str = "matching"


# ---------------------------------------------------------------------------
# Matching-method kwargs
# ---------------------------------------------------------------------------

_MATCHING_KWARGS = {
    "alpha", "num_matches", "replacement", "standard_deviation",
    "model_type", "match_on", "block_size",
}

_MATCHING_DEFAULTS = {
    "alpha": 0,
    "num_matches": 3,
    "replacement": True,
    "standard_deviation": "average",
    "model_type": "logistic",
    "match_on": "logit",
    "block_size": 2000,
}

# ---------------------------------------------------------------------------
# Ebal-method kwargs
# ---------------------------------------------------------------------------

_EBAL_KWARGS = {"moment", "max_weight"}

_EBAL_DEFAULTS = {
    "moment": 1,
    "max_weight": None,
}



def _validate_kwargs(method: str, kwargs: dict, valid: set, defaults: dict) -> dict:
    """Validate and fill defaults for method-specific kwargs."""
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(
            f"Unknown keyword arguments for method='{method}': {unknown}. "
            f"Valid options: {valid}"
        )
    filled = {**defaults, **kwargs}
    return filled


def _run_matching(
    data: pl.DataFrame,
    treat: str, tm: str, entry: str, id: str,
    covariates: list[str],
    lookback: int,
    verbose: bool,
    **kwargs,
) -> RollmatchResult | None:
    """Run matching pipeline: reduce → score → match → weights → balance."""
    opts = _validate_kwargs("matching", kwargs, _MATCHING_KWARGS, _MATCHING_DEFAULTS)

    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch [matching]: {n_treat} treated, {n_ctrl} controls, "
              f"alpha={opts['alpha']}")

    # Step 1: Reduce
    if verbose:
        print("  Step 1: reduce_data...")
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=covariates)
    if verbose:
        print(f"    {reduced.height} rows after NaN removal")

    if reduced.height == 0:
        if verbose:
            print("  ERROR: No valid rows")
        return None

    # Step 2: Score
    if verbose:
        print("  Step 2: score_data...")
    scored = score_data(
        reduced, covariates, treat,
        model_type=opts["model_type"], match_on=opts["match_on"],
    )

    # Step 3: Match
    if verbose:
        print(f"  Step 3: matching (alpha={opts['alpha']}, "
              f"num_matches={opts['num_matches']})...")
    matches = match_all_periods(
        scored, treat, tm, entry, id,
        alpha=opts["alpha"], num_matches=opts["num_matches"],
        replacement=opts["replacement"],
        standard_deviation=opts["standard_deviation"],
        block_size=opts["block_size"],
    )

    if matches is None or matches.height == 0:
        if verbose:
            print("  No matches found!")
        return None

    n_treated_matched = matches["treat_id"].n_unique()
    n_controls_matched = matches["control_id"].n_unique()
    n_treated_total = scored.filter(pl.col(treat) == 1)[id].n_unique()

    if verbose:
        print(f"    Matched: {matches.height} pairs")
        print(f"    Treated: {n_treated_matched}/{n_treated_total} "
              f"({100*n_treated_matched/n_treated_total:.1f}%)")
        print(f"    Controls: {n_controls_matched}")

    # Step 4: Balance
    if verbose:
        print("  Step 4: balance...")
    balance = compute_balance(scored, matches, treat, id, tm, covariates)

    # Step 5: Weights
    weights = _compute_weights(matches, id, opts["num_matches"])

    if verbose:
        smd_table(balance)

    return RollmatchResult(
        matched_data=matches,
        balance=balance,
        n_treated_total=n_treated_total,
        n_treated_matched=n_treated_matched,
        n_controls_matched=n_controls_matched,
        alpha=opts["alpha"],
        weights=weights,
        method="matching",
    )


def _run_ebal(
    data: pl.DataFrame,
    treat: str, tm: str, entry: str, id: str,
    covariates: list[str],
    lookback: int,
    verbose: bool,
    **kwargs,
) -> RollmatchResult | None:
    """Run entropy balancing pipeline: reduce → per-period ebal → balance.

    Each entry cohort runs ebal independently on the FULL control pool.
    Controls are reused across cohorts (stacked design) — the same
    control gets different weights per cohort. This is standard in
    staggered DiD (Cengiz et al. 2019; Baker et al. 2022).

    Returns stacked per-cohort weights in ``weighted_data`` and a
    collapsed convenience weight in ``weights``.
    """
    opts = _validate_kwargs("ebal", kwargs, _EBAL_KWARGS, _EBAL_DEFAULTS)

    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch [ebal]: {n_treat} treated, {n_ctrl} controls, "
              f"moment={opts['moment']}")

    # Step 1: Reduce (no scoring needed)
    if verbose:
        print("  Step 1: reduce_data...")
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=covariates)
    if verbose:
        print(f"    {reduced.height} rows after NaN removal")

    if reduced.height == 0:
        if verbose:
            print("  ERROR: No valid rows")
        return None

    # Step 2: Per-period entropy balancing on full control pool
    if verbose:
        print(f"  Step 2: entropy balancing (moment={opts['moment']})...")

    time_periods = (
        reduced.filter(pl.col(treat) == 1)[tm].unique().sort().to_list()
    )

    stacked_weights = []  # (tm, id, weight) per cohort
    n_treated_total = reduced.filter(pl.col(treat) == 1)[id].n_unique()
    n_periods_ok = 0

    for t in time_periods:
        t_data = reduced.filter((pl.col(treat) == 1) & (pl.col(tm) == t))
        c_data = reduced.filter((pl.col(treat) == 0) & (pl.col(tm) == t))

        if t_data.height == 0 or c_data.height == 0:
            continue

        result = entropy_balance(
            t_data, c_data, covariates, id,
            moment=opts["moment"], max_weight=opts["max_weight"],
        )

        if result is not None:
            # Tag with cohort period
            cohort_weights = result.with_columns(pl.lit(t).alias(tm))
            stacked_weights.append(cohort_weights)
            n_periods_ok += 1

    if not stacked_weights:
        if verbose:
            print("  Entropy balancing failed for all periods!")
        return None

    # Stacked weighted data: (tm, id, weight) — one row per (cohort, unit)
    weighted_data = pl.concat(stacked_weights).select(tm, id, "weight")

    if verbose:
        print(f"    Balanced {n_periods_ok}/{len(time_periods)} periods")

    # Collapsed convenience weights: average weight per unit across cohorts
    # NOTE: This does NOT guarantee per-period balance. Use weighted_data
    # for cohort-specific analysis.
    weights = weighted_data.group_by(id).agg(
        pl.col("weight").mean()
    )

    n_treated_weighted = weights.join(
        reduced.filter(pl.col(treat) == 1).select(id).unique(),
        on=id, how="semi",
    ).height
    n_controls_weighted = weights.join(
        reduced.filter(pl.col(treat) == 0).select(id).unique(),
        on=id, how="semi",
    ).height

    if verbose:
        print(f"    Treated: {n_treated_weighted}/{n_treated_total}")
        print(f"    Controls weighted: {n_controls_weighted}")

    # Step 3: Per-period balance (the correct diagnostic for stacked ebal)
    if verbose:
        print("  Step 3: balance (per-period weighted)...")

    # Compute per-period balance using stacked weights
    from .balance import balance_by_period_weighted
    agg_balance, detail_balance = balance_by_period_weighted(
        reduced, weighted_data, treat, id, tm, covariates,
    )

    # Also compute pooled balance using collapsed weights (for summary)
    pooled_balance = compute_balance_weighted(reduced, weights, treat, id, covariates)

    if verbose:
        smd_table(pooled_balance)
        if agg_balance.height > 0:
            max_per_period = agg_balance["max_abs_smd"].max()
            print(f"  Per-period max|SMD|: {max_per_period:.4f} "
                  f"(pooled may differ due to cohort aggregation)")

    return RollmatchResult(
        matched_data=None,
        balance=pooled_balance,
        n_treated_total=n_treated_total,
        n_treated_matched=n_treated_weighted,
        n_controls_matched=n_controls_weighted,
        alpha=None,
        weights=weights,
        weighted_data=weighted_data,
        method="ebal",
    )


def _run_callable(
    method_fn: Callable,
    data: pl.DataFrame,
    treat: str, tm: str, entry: str, id: str,
    covariates: list[str],
    lookback: int,
    verbose: bool,
    **kwargs,
) -> RollmatchResult | None:
    """Run user-defined method via the pluggable per-period interface.

    Each cohort independently calls the user function on the full control
    pool (stacked design, same as ebal).
    """
    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch [custom]: {n_treat} treated, {n_ctrl} controls")

    # Step 1: Reduce
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=covariates)

    if reduced.height == 0:
        return None

    # Step 2: Per-period custom method on full control pool
    time_periods = (
        reduced.filter(pl.col(treat) == 1)[tm].unique().sort().to_list()
    )

    stacked_weights = []
    n_treated_total = reduced.filter(pl.col(treat) == 1)[id].n_unique()

    for t in time_periods:
        t_data = reduced.filter((pl.col(treat) == 1) & (pl.col(tm) == t))
        c_data = reduced.filter((pl.col(treat) == 0) & (pl.col(tm) == t))

        if t_data.height == 0 or c_data.height == 0:
            continue

        result = method_fn(t_data, c_data, covariates, id, **kwargs)

        if result is not None:
            cohort_weights = result.with_columns(pl.lit(t).alias(tm))
            stacked_weights.append(cohort_weights)

    if not stacked_weights:
        return None

    weighted_data = pl.concat(stacked_weights).select(tm, id, "weight")

    # Collapsed convenience weights
    weights = weighted_data.group_by(id).agg(pl.col("weight").mean())

    n_treated_weighted = weights.join(
        reduced.filter(pl.col(treat) == 1).select(id).unique(),
        on=id, how="semi",
    ).height
    n_controls_weighted = weights.join(
        reduced.filter(pl.col(treat) == 0).select(id).unique(),
        on=id, how="semi",
    ).height

    balance = compute_balance_weighted(reduced, weights, treat, id, covariates)

    if verbose:
        smd_table(balance)

    return RollmatchResult(
        matched_data=None,
        balance=balance,
        n_treated_total=n_treated_total,
        n_treated_matched=n_treated_weighted,
        n_controls_matched=n_controls_weighted,
        alpha=None,
        weights=weights,
        weighted_data=weighted_data,
        method="custom",
    )


def rollmatch(
    data: pl.DataFrame,
    treat: str,
    tm: str,
    entry: str,
    id: str,
    covariates: list[str],
    lookback: int = 1,
    method: str | Callable = "matching",
    verbose: bool = True,
    **method_kwargs,
) -> RollmatchResult | None:
    """Run the rolling entry matching/weighting pipeline.

    Parameters
    ----------
    data : pl.DataFrame
        Panel data with unit x time observations.
    treat : str
        Binary treatment column (1=treated, 0=control).
    tm : str
        Time period column (integer).
    entry : str
        Entry period column.
    id : str
        Unit identifier column.
    covariates : list[str]
        Covariate column names.
    lookback : int
        Periods to look back from entry for baseline covariates.
    method : str or callable
        Method for constructing weights:

        - ``"matching"``: Propensity score nearest-neighbor matching.
          Kwargs: alpha, num_matches, replacement, standard_deviation,
          model_type, match_on, block_size.
        - ``"ebal"``: Entropy balancing (Hainmueller 2012). Directly
          optimizes control weights for exact per-period covariate
          balance. Each cohort independently weights the full control
          pool (stacked design). Returns per-cohort weights in
          ``weighted_data``. Kwargs: moment (1/2/3), max_weight.
        - callable: User-defined function with signature
          ``fn(treated_data, control_data, covariates, id, **kwargs)``
          returning ``pl.DataFrame`` with columns [id, weight].
    verbose : bool
        Print progress.
    **method_kwargs
        Method-specific keyword arguments (see ``method`` above).

    Returns
    -------
    RollmatchResult or None if method fails.
    """
    if callable(method) and not isinstance(method, str):
        return _run_callable(
            method, data, treat, tm, entry, id, covariates,
            lookback, verbose, **method_kwargs,
        )
    elif method == "matching":
        return _run_matching(
            data, treat, tm, entry, id, covariates,
            lookback, verbose, **method_kwargs,
        )
    elif method == "ebal":
        return _run_ebal(
            data, treat, tm, entry, id, covariates,
            lookback, verbose, **method_kwargs,
        )
    else:
        raise ValueError(
            f"method must be 'matching', 'ebal', or a callable, got {method!r}"
        )
