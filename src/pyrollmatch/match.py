"""
match — Core matching engine using vectorized numpy operations.

Avoids the R bottleneck of materializing the full N_treated × N_controls
cross-product. Instead, processes in blocks using numpy broadcasting
for O(block_size × N_controls) memory per iteration.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result from a single time period's matching."""
    treat_ids: np.ndarray
    control_ids: np.ndarray
    differences: np.ndarray
    time_period: int


def compute_caliper_width(
    treated_scores: np.ndarray,
    control_scores: np.ndarray,
    alpha: float,
    standard_deviation: str = "average",
) -> float:
    """Compute caliper width from alpha and pooled standard deviation.

    Parameters
    ----------
    treated_scores : array
        Propensity scores for treated units.
    control_scores : array
        Propensity scores for control units.
    alpha : float
        Caliper multiplier (0 = no caliper).
    standard_deviation : str
        "average" (default), "weighted", or "none".

    Returns
    -------
    float
        Caliper width. inf if alpha == 0.
    """
    if alpha == 0:
        return np.inf

    var_t = np.var(treated_scores, ddof=1) if len(treated_scores) > 1 else 0.0
    var_c = np.var(control_scores, ddof=1) if len(control_scores) > 1 else 0.0

    if standard_deviation == "average":
        pooled_sd = np.sqrt((var_t + var_c) / 2)
    elif standard_deviation == "weighted":
        n_t, n_c = len(treated_scores), len(control_scores)
        pooled_sd = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
    elif standard_deviation == "none":
        pooled_sd = 1.0
    else:
        raise ValueError(f"standard_deviation must be 'average', 'weighted', or 'none'")

    return alpha * pooled_sd


def match_within_period(
    treated_scores: np.ndarray,
    control_scores: np.ndarray,
    treated_ids: np.ndarray,
    control_ids: np.ndarray,
    caliper_width: float,
    num_matches: int = 3,
    replacement: bool = True,
    block_size: int = 2000,
) -> MatchResult | None:
    """Match treated to controls within a single time period.

    Uses block-vectorized numpy broadcasting to avoid materializing
    the full N_treated × N_controls distance matrix.

    Parameters
    ----------
    treated_scores : array of shape (n_treated,)
    control_scores : array of shape (n_controls,)
    treated_ids : array of shape (n_treated,)
    control_ids : array of shape (n_controls,)
    caliper_width : float
        Maximum allowed score difference.
    num_matches : int
        Number of control matches per treated unit.
    replacement : bool
        If True, controls can match multiple treated within same period.
    block_size : int
        Number of treated units to process at once.

    Returns
    -------
    MatchResult or None if no matches found.
    """
    n_treated = len(treated_scores)
    n_controls = len(control_scores)

    if n_treated == 0 or n_controls == 0:
        return None

    all_treat_ids = []
    all_control_ids = []
    all_diffs = []

    # Track which controls are used (for replacement=False)
    used_controls = set() if not replacement else None

    # Process treated units in blocks for memory efficiency
    for block_start in range(0, n_treated, block_size):
        block_end = min(block_start + block_size, n_treated)
        block_scores = treated_scores[block_start:block_end]  # (B,)
        block_ids = treated_ids[block_start:block_end]

        # Available controls
        if not replacement and used_controls:
            available_mask = np.array([
                cid not in used_controls for cid in control_ids
            ])
            avail_scores = control_scores[available_mask]
            avail_ids = control_ids[available_mask]
        else:
            avail_scores = control_scores
            avail_ids = control_ids

        if len(avail_scores) == 0:
            break

        # Compute distance matrix: (B, N_controls)
        # Using broadcasting: |block_scores[:, None] - avail_scores[None, :]|
        dist_matrix = np.abs(block_scores[:, None] - avail_scores[None, :])

        # Apply caliper
        if np.isfinite(caliper_width):
            dist_matrix[dist_matrix > caliper_width] = np.inf

        # Greedy matching: for each treated, find top-k closest controls
        for i in range(len(block_scores)):
            dists = dist_matrix[i]
            valid_mask = np.isfinite(dists)

            if not valid_mask.any():
                continue

            # Get indices sorted by distance
            valid_indices = np.where(valid_mask)[0]
            sorted_idx = valid_indices[np.argsort(dists[valid_indices])]

            # Take top num_matches
            n_take = min(num_matches, len(sorted_idx))
            for j in range(n_take):
                ctrl_idx = sorted_idx[j]
                ctrl_id = avail_ids[ctrl_idx]

                if not replacement and ctrl_id in used_controls:
                    continue

                all_treat_ids.append(block_ids[i])
                all_control_ids.append(ctrl_id)
                all_diffs.append(dists[ctrl_idx])

                if not replacement:
                    used_controls.add(ctrl_id)

    if not all_treat_ids:
        return None

    return MatchResult(
        treat_ids=np.array(all_treat_ids),
        control_ids=np.array(all_control_ids),
        differences=np.array(all_diffs),
        time_period=0,  # set by caller
    )


def match_all_periods(
    scored_data: pl.DataFrame,
    treat: str,
    tm: str,
    entry: str,
    id: str,
    alpha: float,
    num_matches: int = 3,
    replacement: bool = True,
    standard_deviation: str = "average",
    block_size: int = 2000,
) -> pl.DataFrame | None:
    """Run matching across all time periods.

    Parameters
    ----------
    scored_data : pl.DataFrame
        Output from score_data() with "score" column.
    treat, tm, entry, id : str
        Column names.
    alpha : float
        Caliper multiplier.
    num_matches : int
        Controls per treated.
    replacement : bool
        Allow control reuse within period.
    standard_deviation : str
        Method for pooled SD.
    block_size : int
        Block size for memory management.

    Returns
    -------
    pl.DataFrame with columns: tm, treat_id, control_id, difference
    """
    # Compute global caliper width
    all_treated = scored_data.filter(pl.col(treat) == 1)["score"].to_numpy()
    all_controls = scored_data.filter(pl.col(treat) == 0)["score"].to_numpy()
    caliper_width = compute_caliper_width(
        all_treated, all_controls, alpha, standard_deviation
    )

    # Get unique time periods from treated units
    time_periods = (
        scored_data.filter(pl.col(treat) == 1)[tm].unique().sort().to_list()
    )

    all_matches = []

    for t in time_periods:
        # Get treated and control data at this time period
        t_data = scored_data.filter((pl.col(treat) == 1) & (pl.col(tm) == t))
        c_data = scored_data.filter((pl.col(treat) == 0) & (pl.col(tm) == t))

        if t_data.height == 0 or c_data.height == 0:
            continue

        result = match_within_period(
            treated_scores=t_data["score"].to_numpy(),
            control_scores=c_data["score"].to_numpy(),
            treated_ids=t_data[id].to_numpy(),
            control_ids=c_data[id].to_numpy(),
            caliper_width=caliper_width,
            num_matches=num_matches,
            replacement=replacement,
            block_size=block_size,
        )

        if result is not None:
            result.time_period = t
            all_matches.append(result)

    if not all_matches:
        return None

    # Combine all matches into a DataFrame
    matches_df = pl.DataFrame({
        tm: np.concatenate([[m.time_period] * len(m.treat_ids) for m in all_matches]),
        "treat_id": np.concatenate([m.treat_ids for m in all_matches]),
        "control_id": np.concatenate([m.control_ids for m in all_matches]),
        "difference": np.concatenate([m.differences for m in all_matches]),
    })

    return matches_df
