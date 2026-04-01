"""
core — Main rollmatch orchestration and alpha sweep.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass, field

from .reduce import reduce_data
from .score import score_data
from .match import match_all_periods
from .balance import compute_balance, smd_table


@dataclass
class RollmatchResult:
    """Result from rollmatch."""
    matched_data: pl.DataFrame
    balance: pl.DataFrame
    n_treated_total: int
    n_treated_matched: int
    n_controls_matched: int
    alpha: float
    weights: pl.DataFrame  # id -> weight


def _compute_weights(matches: pl.DataFrame, id: str, num_matches: int) -> pl.DataFrame:
    """Compute matching weights from matched pairs.

    Following R rollmatch convention:
    - treatment_weight = 1 / actual_matches_for_this_treated_unit
    - control_weight = sum of treatment_weights across all treatments
      this control is matched to

    This ensures proper inverse probability weighting when treated units
    have different numbers of matches (e.g., due to tight calipers).
    """
    treat_match_counts = (
        matches.group_by("treat_id").len()
        .rename({"len": "total_matches"})
    )
    matches_with_weights = matches.join(treat_match_counts, on="treat_id")
    matches_with_weights = matches_with_weights.with_columns(
        (1.0 / pl.col("total_matches")).alias("treatment_weight")
    )

    treat_weights = (
        matches.select("treat_id").unique()
        .rename({"treat_id": id})
        .with_columns(pl.lit(1.0).alias("weight"))
    )
    ctrl_weights = (
        matches_with_weights
        .group_by("control_id")
        .agg(pl.col("treatment_weight").sum().alias("weight"))
        .rename({"control_id": id})
    )

    weights = pl.concat([treat_weights, ctrl_weights])
    return weights.group_by(id).agg(pl.col("weight").sum())


def rollmatch(
    data: pl.DataFrame,
    treat: str,
    tm: str,
    entry: str,
    id: str,
    covariates: list[str],
    lookback: int = 1,
    alpha: float = 0,
    num_matches: int = 3,
    replacement: str | bool = True,
    standard_deviation: str = "average",
    model_type: str = "logistic",
    match_on: str = "logit",
    block_size: int = 2000,
    verbose: bool = True,
) -> RollmatchResult | None:
    """Run the full rolling entry matching pipeline.

    Parameters
    ----------
    data : pl.DataFrame
        Panel data with unit × time observations.
    treat : str
        Binary treatment column (1=treated, 0=control).
    tm : str
        Time period column (integer).
    entry : str
        Entry period column. Treatment onset for treated units; null or
        any value > max(tm) for controls.
    id : str
        Unit identifier column.
    covariates : list[str]
        Covariate column names for matching.
    lookback : int
        Periods to look back from entry for baseline covariates.
    alpha : float
        Caliper multiplier (0 = no caliper).
    num_matches : int
        Number of control matches per treated unit.
    replacement : str or bool
        Control reuse policy:

        - ``"unrestricted"`` or ``True``: controls reused freely
          (within and across periods).
        - ``"cross_cohort"`` or ``False``: no reuse within a period,
          allowed across periods. This is consistent with R rollmatch.
        - ``"global_no"``: a control matched at most once across all
          periods. Note: results depend on the order periods are
          processed (earliest first).
    standard_deviation : str
        Method for pooled SD in caliper.
    model_type : str
        Propensity model type ("logistic").
    match_on : str
        Score type ("logit" or "pscore").
    block_size : int
        Block size for memory-efficient matching.
    verbose : bool
        Print progress.

    Returns
    -------
    RollmatchResult or None if matching fails.
    """
    if verbose:
        n_treat = data.filter(pl.col(treat) == 1)[id].n_unique()
        n_ctrl = data.filter(pl.col(treat) == 0)[id].n_unique()
        print(f"rollmatch: {n_treat} treated, {n_ctrl} controls, alpha={alpha}")

    # Step 1: Reduce data
    if verbose:
        print("  Step 1: reduce_data...")
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    if verbose:
        print(f"    Reduced: {reduced.height} rows")

    # Drop rows with NaN in covariates
    reduced = reduced.drop_nulls(subset=covariates)
    if verbose:
        print(f"    After dropping NaN: {reduced.height} rows")

    if reduced.height == 0:
        if verbose:
            print("  ERROR: No valid rows after NaN removal")
        return None

    # Step 2: Score data
    if verbose:
        print("  Step 2: score_data...")
    scored = score_data(reduced, covariates, treat, model_type, match_on)
    if verbose:
        print(f"    Scored: {scored.height} rows")

    # Step 3: Match
    if verbose:
        print(f"  Step 3: matching (alpha={alpha}, num_matches={num_matches})...")
    matches = match_all_periods(
        scored, treat, tm, entry, id,
        alpha=alpha, num_matches=num_matches,
        replacement=replacement, standard_deviation=standard_deviation,
        block_size=block_size,
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
        print(f"    Treated matched: {n_treated_matched}/{n_treated_total} "
              f"({100*n_treated_matched/n_treated_total:.1f}%)")
        print(f"    Controls used: {n_controls_matched}")

    # Step 4: Balance
    if verbose:
        print("  Step 4: balance...")
    balance = compute_balance(scored, matches, treat, id, tm, covariates)

    # Step 5: Compute weights
    weights = _compute_weights(matches, id, num_matches)

    if verbose:
        smd_table(balance)

    return RollmatchResult(
        matched_data=matches,
        balance=balance,
        n_treated_total=n_treated_total,
        n_treated_matched=n_treated_matched,
        n_controls_matched=n_controls_matched,
        alpha=alpha,
        weights=weights,
    )


def alpha_sweep(
    data: pl.DataFrame,
    treat: str,
    tm: str,
    entry: str,
    id: str,
    covariates: list[str],
    alphas: list[float] | None = None,
    lookback: int = 1,
    num_matches: int = 3,
    replacement: str | bool = True,
    standard_deviation: str = "average",
    model_type: str = "logistic",
    match_on: str = "logit",
    block_size: int = 2000,
    smd_threshold: float = 0.1,
) -> tuple[pl.DataFrame, RollmatchResult | None]:
    """Run rollmatch across multiple alpha values and select the best.

    Best = fully balanced (all |SMD| < threshold) with highest match rate.
    If none fully balance, select the one with lowest max|SMD|.

    Parameters
    ----------
    data : pl.DataFrame
        Panel data.
    alphas : list[float]
        Caliper multipliers to try. Default: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    smd_threshold : float
        |SMD| threshold for "balanced" (default 0.1).
    (other params same as rollmatch)

    Returns
    -------
    (summary_df, best_result)
    """
    if alphas is None:
        alphas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

    # Pre-compute reduce + score once (shared across alphas)
    reduced = reduce_data(data, treat, tm, entry, id, lookback)
    reduced = reduced.drop_nulls(subset=covariates)
    scored = score_data(reduced, covariates, treat, model_type, match_on)

    results = []
    best_result = None
    best_score = (-1, -np.inf)  # (all_pass, match_rate)

    for alpha in alphas:
        print(f"  alpha={alpha:.2f} ... ", end="", flush=True)

        matches = match_all_periods(
            scored, treat, tm, entry, id,
            alpha=alpha, num_matches=num_matches,
            replacement=replacement, standard_deviation=standard_deviation,
            block_size=block_size,
        )

        if matches is None or matches.height == 0:
            print("no matches")
            continue

        balance = compute_balance(scored, matches, treat, id, tm, covariates)
        max_smd = balance["matched_smd"].abs().max()
        all_pass = max_smd < smd_threshold

        n_treat_total = scored.filter(pl.col(treat) == 1)[id].n_unique()
        n_treat_matched = matches["treat_id"].n_unique()
        match_rate = n_treat_matched / n_treat_total

        results.append({
            "alpha": alpha,
            "n_matched_pairs": matches.height,
            "n_treated_matched": n_treat_matched,
            "pct_treated": round(100 * match_rate, 1),
            "max_abs_smd": round(max_smd, 4),
            "all_pass": all_pass,
        })

        print(f"matched={n_treat_matched}/{n_treat_total} ({100*match_rate:.0f}%), "
              f"max|SMD|={max_smd:.4f} {'✓' if all_pass else '✗'}")

        # Track best
        score = (int(all_pass), match_rate)
        if score > best_score:
            best_score = score
            weights = _compute_weights(matches, id, num_matches)

            best_result = RollmatchResult(
                matched_data=matches,
                balance=balance,
                n_treated_total=n_treat_total,
                n_treated_matched=n_treat_matched,
                n_controls_matched=matches["control_id"].n_unique(),
                alpha=alpha,
                weights=weights,
            )

    summary = pl.DataFrame(results) if results else pl.DataFrame()

    if best_result:
        print(f"\n  Best: alpha={best_result.alpha} "
              f"(matched={best_result.n_treated_matched}/{best_result.n_treated_total}, "
              f"max|SMD|={best_result.balance['matched_smd'].abs().max():.4f})")

    return summary, best_result
