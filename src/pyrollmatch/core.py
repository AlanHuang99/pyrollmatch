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
    """Result from rollmatch."""
    matched_data: pl.DataFrame | None  # None for weighting-only methods
    balance: pl.DataFrame
    n_treated_total: int
    n_treated_matched: int
    n_controls_matched: int
    alpha: float | None                # None for non-caliper methods
    weights: pl.DataFrame              # always [id, weight]
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

# Weighting methods currently require global_no replacement semantics
# (each control assigned to at most one cohort). Supporting cross_cohort
# or unrestricted would require resolving weight aggregation across
# cohorts (stacked vs collapsed structure). See issue #5 discussion.


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
    """Run entropy balancing pipeline: reduce → per-period ebal → balance."""
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

    # Step 2: Per-period entropy balancing with global_no exclusion
    if verbose:
        print(f"  Step 2: entropy balancing (moment={opts['moment']})...")

    time_periods = (
        reduced.filter(pl.col(treat) == 1)[tm].unique().sort().to_list()
    )

    all_weights = []
    used_controls: set = set()
    n_treated_total = reduced.filter(pl.col(treat) == 1)[id].n_unique()

    for t in time_periods:
        t_data = reduced.filter((pl.col(treat) == 1) & (pl.col(tm) == t))
        c_data = reduced.filter((pl.col(treat) == 0) & (pl.col(tm) == t))

        if t_data.height == 0 or c_data.height == 0:
            continue

        # Global no-replacement: exclude already-used controls
        if used_controls:
            c_data = c_data.filter(~pl.col(id).is_in(list(used_controls)))
            if c_data.height == 0:
                continue

        result = entropy_balance(
            t_data, c_data, covariates, id,
            moment=opts["moment"], max_weight=opts["max_weight"],
        )

        if result is not None:
            # Mark controls as used (global_no)
            ctrl_ids = result.filter(pl.col("weight") < 1.0 - 1e-10)[id].to_list()
            # Actually, treated have weight=1.0 and controls have ebal weights
            # Filter to controls: weight != 1.0 (treated all have exactly 1.0)
            ctrl_result = result.join(
                c_data.select(id).unique(), on=id, how="semi"
            )
            used_controls.update(ctrl_result[id].to_list())
            all_weights.append(result)

    if not all_weights:
        if verbose:
            print("  Entropy balancing failed for all periods!")
        return None

    # Combine per-period weights
    # For treated: weight=1.0 (deduplicate if treated appears in multiple periods' results)
    # For controls: each appears once (global_no)
    weights = pl.concat(all_weights).group_by(id).agg(
        pl.col("weight").first()  # treated get 1.0; controls appear once
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

    # Step 3: Balance (weighted)
    if verbose:
        print("  Step 3: balance (weighted)...")
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
    """Run user-defined method via the pluggable per-period interface."""
    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch [custom]: {n_treat} treated, {n_ctrl} controls")

    # Step 1: Reduce
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=covariates)

    if reduced.height == 0:
        return None

    # Step 2: Per-period custom method
    time_periods = (
        reduced.filter(pl.col(treat) == 1)[tm].unique().sort().to_list()
    )

    all_weights = []
    used_controls: set = set()
    n_treated_total = reduced.filter(pl.col(treat) == 1)[id].n_unique()

    for t in time_periods:
        t_data = reduced.filter((pl.col(treat) == 1) & (pl.col(tm) == t))
        c_data = reduced.filter((pl.col(treat) == 0) & (pl.col(tm) == t))

        if t_data.height == 0 or c_data.height == 0:
            continue

        if used_controls:
            c_data = c_data.filter(~pl.col(id).is_in(list(used_controls)))
            if c_data.height == 0:
                continue

        result = method_fn(t_data, c_data, covariates, id, **kwargs)

        if result is not None:
            ctrl_result = result.join(
                c_data.select(id).unique(), on=id, how="semi"
            )
            used_controls.update(ctrl_result[id].to_list())
            all_weights.append(result)

    if not all_weights:
        return None

    weights = pl.concat(all_weights).group_by(id).agg(
        pl.col("weight").first()
    )

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
          optimizes control weights for exact covariate balance.
          Uses global_no replacement (each control in at most one
          cohort). Kwargs: moment (1/2/3), max_weight.
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
