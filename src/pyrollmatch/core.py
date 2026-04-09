"""
core — Main rollmatch orchestration with pluggable method dispatch.

Provides :func:`rollmatch`, the primary entry point for the package. It
orchestrates the reduce → score → match → weight → balance pipeline and
supports three method families:

- **matching**: Nearest-neighbor matching on propensity scores or pairwise
  distances, with optional calipers and replacement control.
- **ebal**: Entropy balancing (Hainmueller 2012) for direct covariate
  balance without propensity scores.
- **callable**: User-defined per-period weighting functions.
"""

from typing import Callable

import polars as pl
import numpy as np
from dataclasses import dataclass

from .reduce import reduce_data
from .score import score_data, DISTANCE_MODELS
from .match import match_all_periods, DistanceSpec
from .weight import _compute_weights, entropy_balance
from .balance import compute_balance, compute_balance_weighted, smd_table


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RollmatchResult:
    """Result from :func:`rollmatch`.

    Attributes
    ----------
    matched_data : pl.DataFrame or None
        Match pairs with columns ``[tm, treat_id, control_id, difference]``.
        ``None`` for weighting-only methods (ebal, custom).
    balance : pl.DataFrame
        Covariate balance table with pre-match and post-match SMDs.
    n_treated_total : int
        Total treated units in the reduced sample.
    n_treated_matched : int
        Treated units that were successfully matched or weighted.
    n_controls_matched : int
        Unique control units used.
    ps_caliper : float or None
        Propensity score caliper multiplier used. ``None`` for non-matching
        methods.
    weights : pl.DataFrame
        Unit-level weights ``[id, weight]``. For matching, derived from
        pairs. For ebal, mean-collapsed across cohorts — an approximation
        that does **not** guarantee exact balance for any individual
        cohort. Prefer ``weighted_data`` for stacked DiD estimation.
    weighted_data : pl.DataFrame or None
        Per-cohort weights ``[tm, id, weight]`` (ebal/custom only).
        **Recommended for downstream analysis**: use with stacked DiD
        (cohort fixed effects) to preserve exact per-cohort balance.
        ``None`` for matching.
    method : str
        Method used: ``"matching"``, ``"ebal"``, or ``"custom"``.
    """
    matched_data: pl.DataFrame | None
    balance: pl.DataFrame
    n_treated_total: int
    n_treated_matched: int
    n_controls_matched: int
    ps_caliper: float | None
    weights: pl.DataFrame
    weighted_data: pl.DataFrame | None = None
    method: str = "matching"


# ---------------------------------------------------------------------------
# Method-specific kwargs
# ---------------------------------------------------------------------------

_MATCHING_KWARGS = {
    "ps_caliper", "num_matches", "replacement", "ps_caliper_std",
    "model_type", "block_size",
    "mahvars", "m_order", "caliper", "std_caliper", "random_state",
}

_MATCHING_DEFAULTS = {
    "ps_caliper": 0,
    "num_matches": 1,
    "replacement": "global_no",
    "ps_caliper_std": "average",
    "model_type": "logistic",
    "block_size": 2000,
    "mahvars": None,
    "m_order": None,
    "caliper": None,
    "std_caliper": True,
    "random_state": None,
}

_EBAL_KWARGS = {"moment", "max_weight"}
_EBAL_DEFAULTS = {"moment": 1, "max_weight": None}


def _validate_kwargs(method: str, kwargs: dict, valid: set, defaults: dict) -> dict:
    """Validate and fill defaults for method-specific kwargs."""
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(
            f"Unknown keyword arguments for method='{method}': {unknown}. "
            f"Valid options: {valid}"
        )
    return {**defaults, **kwargs}


def _validate_treat_column(data: pl.DataFrame, treat: str) -> None:
    """Ensure the treat column is present, numeric, and contains only {0, 1}.

    Raises a clear ValueError rather than letting reduce_data / polars surface
    a low-level ComputeError or silently return an empty reduced dataset.
    """
    if treat not in data.columns:
        raise ValueError(f"treat column '{treat}' not found in data")

    dtype = data.schema[treat]
    if not (dtype.is_integer() or dtype.is_float()):
        raise ValueError(
            f"treat column '{treat}' must be numeric (0/1), got dtype {dtype}. "
            "Recode treated=1 and control=0 before calling rollmatch."
        )

    uniq = data[treat].drop_nulls().unique().to_list()
    bad = [v for v in uniq if v not in (0, 1, 0.0, 1.0)]
    if bad:
        raise ValueError(
            f"treat column '{treat}' must contain only 0 and 1, "
            f"found {sorted(uniq)}. Recode treated=1 and control=0."
        )


def _validate_matching_params(data: pl.DataFrame, opts: dict) -> None:
    """Validate matching-method kwargs up front so users see clear errors
    instead of low-level numpy/polars exceptions or silent None returns."""
    if not isinstance(opts["num_matches"], int) or opts["num_matches"] < 1:
        raise ValueError(
            f"num_matches must be a positive integer, got {opts['num_matches']!r}"
        )
    if not isinstance(opts["block_size"], int) or opts["block_size"] < 1:
        raise ValueError(
            f"block_size must be a positive integer, got {opts['block_size']!r}"
        )
    if opts["ps_caliper"] is not None and opts["ps_caliper"] < 0:
        raise ValueError(
            f"ps_caliper must be >= 0 (0 = no caliper), got {opts['ps_caliper']!r}"
        )
    caliper = opts["caliper"]
    if caliper is not None:
        if not isinstance(caliper, dict):
            raise ValueError(
                f"caliper must be a dict mapping column name to width, "
                f"got {type(caliper).__name__}"
            )
        for var, width in caliper.items():
            if var not in data.columns:
                raise ValueError(
                    f"caliper column '{var}' not found in data"
                )
            if not isinstance(width, (int, float)) or width < 0:
                raise ValueError(
                    f"caliper['{var}'] must be a non-negative number, "
                    f"got {width!r}"
                )


# ---------------------------------------------------------------------------
# Matching pipeline
# ---------------------------------------------------------------------------

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
    _validate_treat_column(data, treat)
    _validate_matching_params(data, opts)

    mahvars = opts["mahvars"]
    model_type = opts["model_type"]
    is_distance = model_type in DISTANCE_MODELS

    # Validate mahvars
    if mahvars is not None:
        if is_distance:
            raise ValueError(
                f"mahvars cannot be used with distance-based model_type='{model_type}'. "
                "Use mahvars with a propensity score model (e.g., 'logistic') to "
                "match on Mahalanobis distance while using PS for caliper."
            )
        for col in mahvars:
            if col not in data.columns:
                raise ValueError(f"mahvars covariate '{col}' not found in data")

    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch [matching]: {n_treat} treated, {n_ctrl} controls, "
              f"ps_caliper={opts['ps_caliper']}")

    # Step 1: Reduce
    if verbose:
        print("  Step 1: reduce_data...")
    all_covs = list(set(covariates + (mahvars or [])))
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=all_covs)
    # Polars drop_nulls doesn't remove NaN; filter those too
    float_covs = [c for c in all_covs if reduced.schema[c].is_float()]
    if float_covs:
        reduced = reduced.filter(
            pl.all_horizontal(~pl.col(c).is_nan() for c in float_covs)
        )
    if verbose:
        print(f"    {reduced.height} rows after NaN removal")
    if reduced.height == 0:
        if verbose:
            print("  ERROR: No valid rows")
        return None
    n_classes = reduced[treat].n_unique()
    if n_classes < 2:
        if verbose:
            print("  ERROR: Need both treated and control units after filtering")
        return None

    # Step 2: Score
    if verbose:
        print("  Step 2: score_data...")
    # When the user sets random_state on rollmatch(), propagate it into the
    # scoring step so stochastic classifiers (rf/gbm/lasso/elasticnet) honor
    # it. When random_state is None (default), score_data uses its own
    # historical default (42) for backward compatibility.
    score_kwargs = {"model_type": model_type}
    if opts["random_state"] is not None:
        score_kwargs["random_state"] = opts["random_state"]
    scored_result = score_data(reduced, covariates, treat, **score_kwargs)
    scored = scored_result.data

    # Step 3: Build DistanceSpec
    dist_spec = DistanceSpec(
        metric=scored_result.distance_metric,
        cov_inv=scored_result.cov_inv,
        transform=scored_result.distance_transform,
    )

    if is_distance:
        dist_spec.covariates = (
            scored_result.ranked_covariates
            if scored_result.ranked_covariates is not None
            else covariates
        )
    elif mahvars is not None:
        # mahvars: PS for caliper, Mahalanobis on mahvars for matching
        from .score import _pooled_within_group_cov
        X_mah = reduced.select(mahvars).to_numpy().astype(np.float64)
        y_mah = reduced[treat].to_numpy().astype(np.int32)
        cov_mah = _pooled_within_group_cov(X_mah, y_mah)
        cov_mah += np.eye(cov_mah.shape[0]) * 1e-6
        dist_spec.metric = "mahalanobis"
        dist_spec.cov_inv = np.linalg.inv(cov_mah)
        dist_spec.covariates = mahvars
        dist_spec.is_mahvars = True

    # Step 4: Match
    if verbose:
        print(f"  Step 3: matching (ps_caliper={opts['ps_caliper']}, "
              f"num_matches={opts['num_matches']})...")
    matches = match_all_periods(
        scored, treat, tm, entry, id,
        ps_caliper=opts["ps_caliper"],
        num_matches=opts["num_matches"],
        replacement=opts["replacement"],
        ps_caliper_std=opts["ps_caliper_std"],
        block_size=opts["block_size"],
        dist_spec=dist_spec,
        m_order=opts["m_order"],
        caliper=opts["caliper"],
        std_caliper=opts["std_caliper"],
        random_state=opts["random_state"],
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

    # Step 5: Weights & balance
    if verbose:
        print("  Step 4: balance...")
    weights = _compute_weights(matches, id, opts["num_matches"])

    # Use weighted balance when weights are non-trivial (replacement with
    # reuse or num_matches > 1); unweighted when all weights are 1.0.
    all_ones = (weights["weight"] - 1.0).abs().max() < 1e-9
    if all_ones:
        balance = compute_balance(scored, matches, treat, id, tm, covariates)
    else:
        balance = compute_balance_weighted(scored, weights, treat, id, covariates)

    if verbose:
        smd_table(balance)

    return RollmatchResult(
        matched_data=matches,
        balance=balance,
        n_treated_total=n_treated_total,
        n_treated_matched=n_treated_matched,
        n_controls_matched=n_controls_matched,
        ps_caliper=opts["ps_caliper"],
        weights=weights,
        method="matching",
    )


# ---------------------------------------------------------------------------
# Entropy balancing pipeline
# ---------------------------------------------------------------------------

def _run_ebal(
    data: pl.DataFrame,
    treat: str, tm: str, entry: str, id: str,
    covariates: list[str],
    lookback: int,
    verbose: bool,
    **kwargs,
) -> RollmatchResult | None:
    """Run entropy balancing: reduce → per-period ebal → balance.

    Stacked design: each cohort weights the full control pool independently.
    """
    opts = _validate_kwargs("ebal", kwargs, _EBAL_KWARGS, _EBAL_DEFAULTS)
    _validate_treat_column(data, treat)

    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch [ebal]: {n_treat} treated, {n_ctrl} controls, "
              f"moment={opts['moment']}")

    if verbose:
        print("  Step 1: reduce_data...")
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=covariates)
    float_covs = [c for c in covariates if reduced.schema[c].is_float()]
    if float_covs:
        reduced = reduced.filter(
            pl.all_horizontal(~pl.col(c).is_nan() for c in float_covs)
        )
    if verbose:
        print(f"    {reduced.height} rows after NaN removal")
    if reduced.height == 0:
        if verbose:
            print("  ERROR: No valid rows")
        return None

    if verbose:
        print(f"  Step 2: entropy balancing (moment={opts['moment']})...")

    time_periods = (
        reduced.filter(pl.col(treat) == 1)[tm].unique().sort().to_list()
    )

    stacked_weights = []
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
            stacked_weights.append(result.with_columns(pl.lit(t).alias(tm)))
            n_periods_ok += 1

    if not stacked_weights:
        if verbose:
            print("  Entropy balancing failed for all periods!")
        return None

    weighted_data = pl.concat(stacked_weights).select(tm, id, "weight")
    if verbose:
        print(f"    Balanced {n_periods_ok}/{len(time_periods)} periods")

    weights = weighted_data.group_by(id).agg(pl.col("weight").mean())

    n_treated_weighted = weights.join(
        reduced.filter(pl.col(treat) == 1).select(id).unique(), on=id, how="semi",
    ).height
    n_controls_weighted = weights.join(
        reduced.filter(pl.col(treat) == 0).select(id).unique(), on=id, how="semi",
    ).height

    if verbose:
        print(f"    Treated: {n_treated_weighted}/{n_treated_total}")
        print(f"    Controls weighted: {n_controls_weighted}")
        print("  Step 3: balance (per-period weighted)...")

    from .balance import balance_by_period_weighted
    agg_balance, _ = balance_by_period_weighted(
        reduced, weighted_data, treat, id, tm, covariates,
    )
    pooled_balance = compute_balance_weighted(reduced, weights, treat, id, covariates)

    if verbose:
        smd_table(pooled_balance)
        if agg_balance.height > 0:
            max_per_period = agg_balance["max_abs_smd"].max()
            print(f"  Per-period max|SMD|: {max_per_period:.4f}")

    return RollmatchResult(
        matched_data=None,
        balance=pooled_balance,
        n_treated_total=n_treated_total,
        n_treated_matched=n_treated_weighted,
        n_controls_matched=n_controls_weighted,
        ps_caliper=None,
        weights=weights,
        weighted_data=weighted_data,
        method="ebal",
    )


# ---------------------------------------------------------------------------
# Custom callable pipeline
# ---------------------------------------------------------------------------

def _run_callable(
    method_fn: Callable,
    data: pl.DataFrame,
    treat: str, tm: str, entry: str, id: str,
    covariates: list[str],
    lookback: int,
    verbose: bool,
    **kwargs,
) -> RollmatchResult | None:
    """Run user-defined per-period weighting function (stacked design)."""
    _validate_treat_column(data, treat)
    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch [custom]: {n_treat} treated, {n_ctrl} controls")

    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=covariates)
    float_covs = [c for c in covariates if reduced.schema[c].is_float()]
    if float_covs:
        reduced = reduced.filter(
            pl.all_horizontal(~pl.col(c).is_nan() for c in float_covs)
        )
    if reduced.height == 0:
        return None

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
            stacked_weights.append(result.with_columns(pl.lit(t).alias(tm)))

    if not stacked_weights:
        return None

    weighted_data = pl.concat(stacked_weights).select(tm, id, "weight")
    weights = weighted_data.group_by(id).agg(pl.col("weight").mean())

    n_treated_weighted = weights.join(
        reduced.filter(pl.col(treat) == 1).select(id).unique(), on=id, how="semi",
    ).height
    n_controls_weighted = weights.join(
        reduced.filter(pl.col(treat) == 0).select(id).unique(), on=id, how="semi",
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
        ps_caliper=None,
        weights=weights,
        weighted_data=weighted_data,
        method="custom",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    This is the main entry point for pyrollmatch. It orchestrates data
    reduction, scoring, matching (or weighting), and balance computation
    in a single call.

    Parameters
    ----------
    data : pl.DataFrame
        Panel data with unit × time observations. Must contain columns
        for treatment status, time period, entry period, unit ID, and
        covariates.
    treat : str
        Column name for binary treatment indicator (1=treated, 0=control).
    tm : str
        Column name for time period (integer).
    entry : str
        Column name for entry period. Treated units: the period when
        treatment begins. Controls: any value > max(tm) or null.
    id : str
        Column name for unit identifier.
    covariates : list[str]
        Covariate column names used for scoring and balance diagnostics.
    lookback : int, default 1
        Number of periods before entry to use as baseline. Must be >= 1.
    method : str or callable, default ``"matching"``
        Weighting method:

        ``"matching"``
            Nearest-neighbor matching on propensity scores or pairwise
            distances. Returns match pairs in ``matched_data``.

            **Matching kwargs:**

            - ``ps_caliper`` (float, default 0): PS caliper multiplier.
              0 = no caliper.
            - ``ps_caliper_std`` (str, default ``"average"``): How to
              compute pooled SD for PS caliper. ``"average"``,
              ``"weighted"``, or ``"none"``.
            - ``num_matches`` (int, default 1): Controls per treated.
            - ``replacement`` (str, default ``"global_no"``):
              ``"global_no"``, ``"cross_cohort"``, or ``"unrestricted"``.
            - ``model_type`` (str, default ``"logistic"``): Scoring model.
              See :data:`~pyrollmatch.score.SUPPORTED_MODELS`.
            - ``block_size`` (int, default 2000): Block size for memory.
            - ``mahvars`` (list[str] or None): Covariates for Mahalanobis
              distance matching with PS caliper (MatchIt pattern).
            - ``m_order`` (str or None): Matching order: ``"largest"``,
              ``"smallest"``, ``"random"``, ``"data"``, or ``None`` (auto).
            - ``caliper`` (dict or None): Per-variable calipers,
              e.g. ``{"x1": 0.5, "x2": 0.3}``.
            - ``std_caliper`` (bool, default True): Whether per-variable
              caliper widths are in SD units.
            - ``random_state`` (int or None, default None): Seed for
              reproducible matching when ``m_order="random"``.

        ``"ebal"``
            Entropy balancing (Hainmueller 2012). Returns per-cohort
            weights in ``weighted_data`` (recommended for stacked DiD
            estimation with cohort fixed effects) and mean-collapsed
            ``weights`` (approximate, for single-panel estimators).

            **Ebal kwargs:**

            - ``moment`` (int, default 1): Moment constraint (1=mean,
              2=mean+variance, 3=mean+variance+skewness).
            - ``max_weight`` (float or None): Maximum weight cap.
              Capping may violate exact moment balance; a warning is
              emitted if balance degrades significantly.

        callable
            User-defined function with signature
            ``fn(treated_data, control_data, covariates, id, **kwargs)``
            returning ``pl.DataFrame`` with columns ``[id, weight]``.

    verbose : bool, default True
        Print progress and balance summary.
    **method_kwargs
        Method-specific keyword arguments (see ``method`` above).

    Returns
    -------
    RollmatchResult or None
        Contains ``matched_data``, ``balance``, ``weights``, and summary
        statistics. Returns ``None`` if no matches/weights could be
        computed (e.g. empty data, convergence failure).

    Examples
    --------
    Propensity score matching (default):

    >>> result = rollmatch(
    ...     data, treat="treat", tm="time", entry="entry_time",
    ...     id="unit_id", covariates=["x1", "x2", "x3"],
    ...     ps_caliper=0.2, num_matches=3,
    ... )
    >>> result.matched_data    # match pairs
    >>> result.balance         # SMD table

    Mahalanobis distance matching:

    >>> result = rollmatch(
    ...     data, treat="treat", tm="time", entry="entry_time",
    ...     id="unit_id", covariates=["x1", "x2", "x3"],
    ...     model_type="mahalanobis",
    ... )

    Mahalanobis matching with PS caliper (MatchIt ``mahvars`` pattern):

    >>> result = rollmatch(
    ...     data, treat="treat", tm="time", entry="entry_time",
    ...     id="unit_id", covariates=["x1", "x2", "x3"],
    ...     ps_caliper=0.25, mahvars=["x1", "x2"],
    ... )

    Entropy balancing:

    >>> result = rollmatch(
    ...     data, treat="treat", tm="time", entry="entry_time",
    ...     id="unit_id", covariates=["x1", "x2", "x3"],
    ...     method="ebal", moment=1,
    ... )
    >>> result.weights         # unit weights
    >>> result.weighted_data   # per-cohort weights
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
