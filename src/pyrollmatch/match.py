"""
match — Core matching engine using vectorized numpy operations.

Provides nearest-neighbor greedy matching within rolling entry time periods.
Supports propensity score matching and pairwise distance matching
(Mahalanobis, Euclidean, scaled Euclidean, robust Mahalanobis).

Architecture
------------
``match_all_periods`` iterates over time periods and calls
``match_within_period`` for each. The inner function processes treated
units in blocks of ``block_size`` to bound memory at
O(block_size × N_controls) per iteration, rather than materializing the
full N_treated × N_controls matrix.

Distance computation is delegated to ``_compute_distance_matrix``, which
dispatches on the ``DistanceSpec.metric`` field.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Result from a single time period's matching.

    Attributes
    ----------
    treat_ids : np.ndarray
        Matched treated unit IDs.
    control_ids : np.ndarray
        Matched control unit IDs (same length as ``treat_ids``).
    differences : np.ndarray
        Distance/score difference for each pair.
    time_period : int
        The time period this result belongs to.
    """
    treat_ids: np.ndarray
    control_ids: np.ndarray
    differences: np.ndarray
    time_period: int


@dataclass
class DistanceSpec:
    """Bundle of distance-related parameters for the matching engine.

    Encapsulates all metadata needed to compute pairwise distances
    between treated and control units. For propensity score matching,
    all fields are ``None``/default and the matcher uses scalar score
    differences.

    Attributes
    ----------
    metric : str or None
        Distance metric: ``"mahalanobis"``, ``"robust_mahalanobis"``,
        ``"scaled_euclidean"``, ``"euclidean"``, or ``None`` for
        propensity score matching.
    covariates : list[str] or None
        Covariate column names to extract from the scored DataFrame
        for pairwise distance computation.
    cov_inv : np.ndarray or None
        Inverse covariance matrix (for Mahalanobis variants).
    transform : np.ndarray or None
        Diagonal scaling matrix (for scaled Euclidean).
    is_mahvars : bool
        If True, propensity scores are real (mahvars pattern) and the
        PS caliper should be applied alongside distance matching.
    """
    metric: str | None = None
    covariates: list[str] | None = None
    cov_inv: np.ndarray | None = None
    transform: np.ndarray | None = None
    is_mahvars: bool = False

    @property
    def use_pairwise(self) -> bool:
        """Whether to compute pairwise covariate distances."""
        return self.metric is not None and self.covariates is not None


# ---------------------------------------------------------------------------
# Caliper computation
# ---------------------------------------------------------------------------

def compute_caliper_width(
    treated_scores: np.ndarray,
    control_scores: np.ndarray,
    ps_caliper: float,
    ps_caliper_std: str = "average",
) -> float:
    """Compute propensity score caliper width.

    Parameters
    ----------
    treated_scores : np.ndarray
        Propensity scores for treated units.
    control_scores : np.ndarray
        Propensity scores for control units.
    ps_caliper : float
        Caliper multiplier. 0 means no caliper (returns ``inf``).
    ps_caliper_std : str, default ``"average"``
        How to compute pooled SD for the caliper:

        - ``"average"``: ``sqrt((var_t + var_c) / 2)``
        - ``"weighted"``: pooled SD weighted by group sizes
        - ``"none"``: ``pooled_sd = 1`` (caliper in raw score units)

    Returns
    -------
    float
        Caliper width. ``inf`` if ``ps_caliper == 0``.
    """
    if ps_caliper == 0:
        return np.inf

    var_t = np.var(treated_scores, ddof=1) if len(treated_scores) > 1 else 0.0
    var_c = np.var(control_scores, ddof=1) if len(control_scores) > 1 else 0.0

    if ps_caliper_std == "average":
        pooled_sd = np.sqrt((var_t + var_c) / 2)
    elif ps_caliper_std == "weighted":
        n_t, n_c = len(treated_scores), len(control_scores)
        pooled_sd = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
    elif ps_caliper_std == "none":
        pooled_sd = 1.0
    else:
        raise ValueError(
            f"ps_caliper_std must be 'average', 'weighted', or 'none', "
            f"got {ps_caliper_std!r}"
        )

    return ps_caliper * pooled_sd


# ---------------------------------------------------------------------------
# Replacement mode
# ---------------------------------------------------------------------------

_VALID_REPLACEMENT = {"unrestricted", "cross_cohort", "global_no"}


def _normalize_replacement(replacement: str) -> str:
    """Validate and return the replacement mode string.

    Valid values:

    - ``"unrestricted"``: controls reused freely across all periods.
    - ``"cross_cohort"``: no reuse within a period, allowed across periods.
    - ``"global_no"``: each control matched at most once across all periods.
    """
    if replacement not in _VALID_REPLACEMENT:
        raise ValueError(
            f"replacement must be one of {_VALID_REPLACEMENT}, got {replacement!r}"
        )
    return replacement


# ---------------------------------------------------------------------------
# Distance matrix computation
# ---------------------------------------------------------------------------

def _compute_distance_matrix(
    block_scores: np.ndarray,
    avail_scores: np.ndarray,
    block_covs: np.ndarray | None,
    avail_covs: np.ndarray | None,
    dist_spec: DistanceSpec,
) -> np.ndarray:
    """Compute pairwise distance matrix between a treated block and controls.

    Parameters
    ----------
    block_scores, avail_scores : np.ndarray
        Propensity scores for the block and available controls.
    block_covs, avail_covs : np.ndarray or None
        Covariate matrices for distance-based matching.
    dist_spec : DistanceSpec
        Distance specification.

    Returns
    -------
    np.ndarray, shape (B, N_avail)
    """
    metric = dist_spec.metric

    if metric is None:
        return np.abs(block_scores[:, None] - avail_scores[None, :])

    if metric in ("mahalanobis", "robust_mahalanobis"):
        return cdist(block_covs, avail_covs, metric='mahalanobis', VI=dist_spec.cov_inv)

    if metric == "scaled_euclidean":
        diag = np.diag(dist_spec.transform)
        return cdist(block_covs * diag, avail_covs * diag, metric='euclidean')

    if metric == "euclidean":
        return cdist(block_covs, avail_covs, metric='euclidean')

    raise ValueError(f"Unknown distance metric: {metric}")


# ---------------------------------------------------------------------------
# Matching order
# ---------------------------------------------------------------------------

def _sort_treated_indices(
    n_treated: int,
    treated_scores: np.ndarray,
    m_order: str | None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return indices determining the order in which treated units are matched.

    Parameters
    ----------
    n_treated : int
    treated_scores : np.ndarray, shape (n_treated,)
    m_order : str or None
        ``"largest"`` — descending score (default for PS; matches hard units
        first). ``"smallest"`` — ascending. ``"random"`` — random permutation.
        ``"data"`` — original order. ``None`` — auto-detect (``"largest"``
        if scores vary, ``"data"`` for distance models).
    rng : np.random.Generator or None

    Returns
    -------
    np.ndarray, shape (n_treated,)
        Permutation indices.
    """
    if m_order is None:
        if np.all(treated_scores == treated_scores[0]):
            m_order = "data"
        else:
            m_order = "largest"

    if m_order == "data":
        return np.arange(n_treated)
    elif m_order == "largest":
        return np.argsort(-treated_scores)
    elif m_order == "smallest":
        return np.argsort(treated_scores)
    elif m_order == "random":
        if rng is None:
            rng = np.random.default_rng()
        return rng.permutation(n_treated)
    else:
        raise ValueError(
            f"m_order must be 'largest', 'smallest', 'random', or 'data', "
            f"got {m_order!r}"
        )


# ---------------------------------------------------------------------------
# Fast path: scalar propensity-score matching
# ---------------------------------------------------------------------------

def _match_propensity_sorted(
    treated_scores: np.ndarray,
    control_scores: np.ndarray,
    treated_ids: np.ndarray,
    control_ids: np.ndarray,
    caliper_width: float,
    num_matches: int,
    replacement: str,
    _used_controls: set | None,
    m_order: str | None,
    rng: np.random.Generator | None,
) -> MatchResult | None:
    """Match on scalar scores using sorted controls instead of all pairs.

    For one-dimensional propensity-score matching, the nearest controls to a
    treated score must lie around its insertion point in sorted control-score
    order. This avoids materializing a ``n_treated × n_controls`` distance
    matrix for the common propensity-score path.
    """
    mode = _normalize_replacement(replacement)
    n_treated = len(treated_scores)
    n_controls = len(control_scores)

    if n_treated == 0 or n_controls == 0:
        return None

    max_matches = n_treated * num_matches
    out_treat = np.empty(max_matches, dtype=treated_ids.dtype)
    out_ctrl = np.empty(max_matches, dtype=control_ids.dtype)
    out_diff = np.empty(max_matches, dtype=np.float64)
    n_matched = 0

    order = _sort_treated_indices(n_treated, treated_scores, m_order, rng=rng)
    treated_scores = treated_scores[order]
    treated_ids = treated_ids[order]

    control_sort_order = np.argsort(control_scores, kind="mergesort")
    sorted_scores = control_scores[control_sort_order]

    if mode == "unrestricted":
        available = None
        used_controls = None
    else:
        available = np.ones(n_controls, dtype=bool)
        if mode == "global_no":
            used_controls = _used_controls if _used_controls is not None else set()
            if used_controls:
                used_arr = np.fromiter(
                    used_controls, dtype=control_ids.dtype, count=len(used_controls)
                )
                available[np.isin(control_ids, used_arr, assume_unique=True)] = False
        else:
            used_controls = set()

    caliper_is_finite = np.isfinite(caliper_width)

    for i in range(n_treated):
        score = treated_scores[i]
        right = int(np.searchsorted(sorted_scores, score, side="left"))
        left = right - 1
        n_take = 0

        while n_take < num_matches and (left >= 0 or right < n_controls):
            if available is not None:
                while left >= 0 and not available[control_sort_order[left]]:
                    left -= 1
                while right < n_controls and not available[control_sort_order[right]]:
                    right += 1

            left_dist = abs(score - sorted_scores[left]) if left >= 0 else np.inf
            right_dist = abs(sorted_scores[right] - score) if right < n_controls else np.inf

            if left_dist <= right_dist:
                dist = left_dist
                sorted_pos = left
                left -= 1
            else:
                dist = right_dist
                sorted_pos = right
                right += 1

            if not np.isfinite(dist):
                break
            if caliper_is_finite and dist > caliper_width:
                break

            control_pos = control_sort_order[sorted_pos]
            out_treat[n_matched] = treated_ids[i]
            out_ctrl[n_matched] = control_ids[control_pos]
            out_diff[n_matched] = dist
            n_matched += 1
            n_take += 1

            if available is not None:
                available[control_pos] = False
                if used_controls is not None:
                    used_controls.add(control_ids[control_pos])

    if n_matched == 0:
        return None

    return MatchResult(
        treat_ids=out_treat[:n_matched],
        control_ids=out_ctrl[:n_matched],
        differences=out_diff[:n_matched],
        time_period=0,
    )


# ---------------------------------------------------------------------------
# Single-period matching
# ---------------------------------------------------------------------------

def match_within_period(
    treated_scores: np.ndarray,
    control_scores: np.ndarray,
    treated_ids: np.ndarray,
    control_ids: np.ndarray,
    caliper_width: float,
    num_matches: int = 1,
    replacement: str = "cross_cohort",
    block_size: int = 2000,
    _used_controls: set | None = None,
    treated_covs: np.ndarray | None = None,
    control_covs: np.ndarray | None = None,
    dist_spec: DistanceSpec | None = None,
    m_order: str | None = None,
    var_caliper_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> MatchResult | None:
    """Match treated to controls within a single time period.

    Uses greedy nearest-neighbor matching with block-vectorized distance
    computation. For each treated unit (in ``m_order``), finds the closest
    ``num_matches`` controls that satisfy caliper constraints and haven't
    been used (if replacement is constrained).

    Parameters
    ----------
    treated_scores, control_scores : np.ndarray, shape (n,)
        Propensity scores (or zeros for distance-based matching).
    treated_ids, control_ids : np.ndarray, shape (n,)
        Unit identifiers.
    caliper_width : float
        Maximum allowed PS distance. ``inf`` for no caliper.
    num_matches : int, default 1
        Controls per treated unit.
    replacement : str, default ``"cross_cohort"``
        ``"unrestricted"``, ``"cross_cohort"``, or ``"global_no"``.
    block_size : int, default 2000
        Treated units per block (memory/speed tradeoff).
    _used_controls : set or None
        Externally managed set for ``"global_no"`` mode.
    treated_covs, control_covs : np.ndarray or None
        Covariate matrices for distance-based matching.
    dist_spec : DistanceSpec or None
        Distance specification. ``None`` for propensity score matching.
    m_order : str or None
        Matching order for treated units.
    var_caliper_mask : np.ndarray or None, shape (n_treated, n_controls)
        Boolean mask from per-variable calipers. ``True`` = pair allowed.

    Returns
    -------
    MatchResult or None
        ``None`` if no matches found.
    """
    mode = _normalize_replacement(replacement)
    if dist_spec is None:
        dist_spec = DistanceSpec()

    n_treated = len(treated_scores)
    n_controls = len(control_scores)

    if n_treated == 0 or n_controls == 0:
        return None

    if dist_spec.metric is None and var_caliper_mask is None:
        return _match_propensity_sorted(
            treated_scores=treated_scores,
            control_scores=control_scores,
            treated_ids=treated_ids,
            control_ids=control_ids,
            caliper_width=caliper_width,
            num_matches=num_matches,
            replacement=mode,
            _used_controls=_used_controls,
            m_order=m_order,
            rng=rng,
        )

    use_pairwise = dist_spec.use_pairwise and treated_covs is not None

    # Pre-allocate output arrays
    max_matches = n_treated * num_matches
    out_treat = np.empty(max_matches, dtype=treated_ids.dtype)
    out_ctrl = np.empty(max_matches, dtype=control_ids.dtype)
    out_diff = np.empty(max_matches, dtype=np.float64)
    n_matched = 0

    # Track used controls via incremental boolean mask
    if mode == "unrestricted":
        available = None
        used_controls = None
    else:
        available = np.ones(n_controls, dtype=bool)
        if mode == "global_no":
            used_controls = _used_controls if _used_controls is not None else set()
            if used_controls:
                used_arr = np.fromiter(used_controls, dtype=control_ids.dtype,
                                       count=len(used_controls))
                available[np.isin(control_ids, used_arr, assume_unique=True)] = False
        else:
            used_controls = set()

    # Reverse index: control_id → position (via sorted searchsorted)
    ctrl_sort_order = np.argsort(control_ids)
    ctrl_ids_sorted = control_ids[ctrl_sort_order]

    def _ctrl_id_to_idx(cid):
        return ctrl_sort_order[np.searchsorted(ctrl_ids_sorted, cid)]

    # Determine matching order
    order = _sort_treated_indices(n_treated, treated_scores, m_order, rng=rng)
    treated_scores = treated_scores[order]
    treated_ids = treated_ids[order]
    if treated_covs is not None:
        treated_covs = treated_covs[order]
    if var_caliper_mask is not None:
        var_caliper_mask = var_caliper_mask[order]

    # Process treated in blocks
    for block_start in range(0, n_treated, block_size):
        block_end = min(block_start + block_size, n_treated)
        block_scores = treated_scores[block_start:block_end]
        block_ids = treated_ids[block_start:block_end]
        block_covs = treated_covs[block_start:block_end] if use_pairwise else None
        block_var_mask = (var_caliper_mask[block_start:block_end]
                         if var_caliper_mask is not None else None)

        # Get available controls
        if available is not None:
            avail_indices = np.where(available)[0]
            avail_scores = control_scores[avail_indices]
            avail_ids = control_ids[avail_indices]
            avail_covs = control_covs[avail_indices] if use_pairwise else None
        else:
            avail_indices = None
            avail_scores = control_scores
            avail_ids = control_ids
            avail_covs = control_covs if use_pairwise else None

        if len(avail_ids) == 0:
            break

        # Distance matrix: (B, N_avail)
        dist_matrix = _compute_distance_matrix(
            block_scores, avail_scores,
            block_covs, avail_covs, dist_spec,
        )

        # Apply PS caliper
        if np.isfinite(caliper_width):
            if dist_spec.metric is not None and dist_spec.is_mahvars:
                # mahvars: caliper on PS, not matching distance
                ps_dist = np.abs(block_scores[:, None] - avail_scores[None, :])
                dist_matrix[ps_dist > caliper_width] = np.inf
            elif dist_spec.metric is None:
                dist_matrix[dist_matrix > caliper_width] = np.inf

        # Apply per-variable caliper mask
        if block_var_mask is not None:
            if avail_indices is not None:
                block_var_avail = block_var_mask[:, avail_indices]
            else:
                block_var_avail = block_var_mask
            dist_matrix[~block_var_avail] = np.inf

        # Greedy matching with argpartition for O(N) top-k
        n_block = len(block_scores)
        if available is None:
            # Unrestricted: no sequential dependency
            for i in range(n_block):
                dists = dist_matrix[i]
                finite_mask = np.isfinite(dists)
                n_finite = finite_mask.sum()
                if n_finite == 0:
                    continue
                finite_idx = np.where(finite_mask)[0]
                n_take = min(num_matches, n_finite)
                if n_take < n_finite:
                    part_idx = np.argpartition(dists[finite_idx], n_take)[:n_take]
                    top_idx = finite_idx[part_idx]
                    top_idx = top_idx[np.argsort(dists[top_idx])]
                else:
                    top_idx = finite_idx[np.argsort(dists[finite_idx])]
                end = n_matched + n_take
                out_treat[n_matched:end] = block_ids[i]
                out_ctrl[n_matched:end] = avail_ids[top_idx]
                out_diff[n_matched:end] = dists[top_idx]
                n_matched = end
        else:
            # Constrained: sequential with within-block invalidation
            for i in range(n_block):
                dists = dist_matrix[i]
                finite_mask = np.isfinite(dists)
                n_finite = finite_mask.sum()
                if n_finite == 0:
                    continue
                finite_idx = np.where(finite_mask)[0]
                n_take = min(num_matches, n_finite)
                if n_take < n_finite:
                    part_idx = np.argpartition(dists[finite_idx], n_take)[:n_take]
                    top_idx = finite_idx[part_idx]
                    top_idx = top_idx[np.argsort(dists[top_idx])]
                else:
                    top_idx = finite_idx[np.argsort(dists[finite_idx])]
                for j in range(n_take):
                    ci = top_idx[j]
                    cid = avail_ids[ci]
                    out_treat[n_matched] = block_ids[i]
                    out_ctrl[n_matched] = cid
                    out_diff[n_matched] = dists[ci]
                    n_matched += 1
                    available[_ctrl_id_to_idx(cid)] = False
                    dist_matrix[i+1:, ci] = np.inf
                    if used_controls is not None:
                        used_controls.add(cid)

    if n_matched == 0:
        return None

    return MatchResult(
        treat_ids=out_treat[:n_matched],
        control_ids=out_ctrl[:n_matched],
        differences=out_diff[:n_matched],
        time_period=0,
    )


# ---------------------------------------------------------------------------
# Multi-period matching
# ---------------------------------------------------------------------------

def match_all_periods(
    scored_data: pl.DataFrame,
    treat: str,
    tm: str,
    entry: str,
    id: str,
    ps_caliper: float,
    num_matches: int = 1,
    replacement: str = "cross_cohort",
    ps_caliper_std: str = "average",
    block_size: int = 2000,
    dist_spec: DistanceSpec | None = None,
    m_order: str | None = None,
    caliper: dict[str, float] | None = None,
    std_caliper: bool = True,
    random_state: int | None = None,
) -> pl.DataFrame | None:
    """Run nearest-neighbor matching across all time periods.

    For each entry cohort (time period), matches treated units to controls
    using the specified distance metric and caliper constraints.

    Parameters
    ----------
    scored_data : pl.DataFrame
        Output from :func:`score_data` (via ``.data``), with ``"score"`` column.
    treat, tm, entry, id : str
        Column names for treatment indicator, time period, entry period,
        and unit identifier.
    ps_caliper : float
        Propensity score caliper multiplier. 0 means no caliper.
    num_matches : int, default 1
        Number of control matches per treated unit.
    replacement : str, default ``"cross_cohort"``
        Control reuse policy:

        - ``"unrestricted"``: controls reused freely.
        - ``"cross_cohort"``: no reuse within a period, allowed across.
        - ``"global_no"``: each control matched at most once globally.
    ps_caliper_std : str, default ``"average"``
        How to compute the pooled SD for PS caliper width.
    block_size : int, default 2000
        Block size for memory management.
    dist_spec : DistanceSpec or None
        Distance specification for pairwise matching. ``None`` for PS matching.
    m_order : str or None
        Matching order. See :func:`_sort_treated_indices`.
    caliper : dict[str, float] or None
        Per-variable calipers: ``{var_name: width}``.
    std_caliper : bool, default True
        If True, per-variable caliper widths are in SD units.

    Returns
    -------
    pl.DataFrame or None
        Columns: ``[tm, treat_id, control_id, difference]``.
        ``None`` if no matches found.
    """
    mode = _normalize_replacement(replacement)
    if dist_spec is None:
        dist_spec = DistanceSpec()

    rng = np.random.default_rng(random_state) if random_state is not None else None

    # Compute global PS caliper width
    if dist_spec.use_pairwise and not dist_spec.is_mahvars:
        caliper_width = np.inf  # pure distance: no PS caliper
    else:
        all_treated = scored_data.filter(pl.col(treat) == 1)["score"].to_numpy()
        all_controls = scored_data.filter(pl.col(treat) == 0)["score"].to_numpy()
        caliper_width = compute_caliper_width(
            all_treated, all_controls, ps_caliper, ps_caliper_std
        )

    # --- Extract all numpy arrays ONCE ---
    all_treat_mask = scored_data[treat].to_numpy() == 1
    all_scores = scored_data["score"].to_numpy()
    all_ids = scored_data[id].to_numpy()
    all_tm = scored_data[tm].to_numpy()

    all_covs = None
    if dist_spec.covariates is not None:
        all_covs = scored_data.select(dist_spec.covariates).to_numpy().astype(np.float64)

    # Pre-compute per-variable caliper absolute widths
    caliper_widths_abs = None
    caliper_var_arrays = None
    if caliper is not None:
        caliper_widths_abs = {}
        caliper_var_arrays = {}
        for var, width in caliper.items():
            caliper_var_arrays[var] = scored_data[var].to_numpy().astype(np.float64)
            if std_caliper:
                vals = caliper_var_arrays[var]
                sd_t = np.std(vals[all_treat_mask], ddof=1) if all_treat_mask.sum() > 1 else 1.0
                sd_c = np.std(vals[~all_treat_mask], ddof=1) if (~all_treat_mask).sum() > 1 else 1.0
                pooled_sd = np.sqrt((sd_t**2 + sd_c**2) / 2)
                if pooled_sd < 1e-10:
                    pooled_sd = 1.0
                caliper_widths_abs[var] = width * pooled_sd
            else:
                caliper_widths_abs[var] = width

    # Iterate over time periods
    treated_tm = all_tm[all_treat_mask]
    time_periods = np.unique(treated_tm)
    time_periods.sort()

    all_matches = []
    global_used = set() if mode == "global_no" else None

    for t in time_periods:
        t_idx = np.where(all_treat_mask & (all_tm == t))[0]
        c_idx = np.where(~all_treat_mask & (all_tm == t))[0]
        if len(t_idx) == 0 or len(c_idx) == 0:
            continue

        t_covs = all_covs[t_idx] if all_covs is not None else None
        c_covs = all_covs[c_idx] if all_covs is not None else None

        # Per-variable caliper mask
        var_caliper_mask = None
        if caliper_widths_abs is not None:
            mask = np.ones((len(t_idx), len(c_idx)), dtype=bool)
            for var, abs_width in caliper_widths_abs.items():
                arr = caliper_var_arrays[var]
                mask &= (np.abs(arr[t_idx, None] - arr[c_idx]) <= abs_width)
            var_caliper_mask = mask

        result = match_within_period(
            treated_scores=all_scores[t_idx],
            control_scores=all_scores[c_idx],
            treated_ids=all_ids[t_idx],
            control_ids=all_ids[c_idx],
            caliper_width=caliper_width,
            num_matches=num_matches,
            replacement=mode,
            block_size=block_size,
            _used_controls=global_used,
            treated_covs=t_covs,
            control_covs=c_covs,
            dist_spec=dist_spec,
            m_order=m_order,
            var_caliper_mask=var_caliper_mask,
            rng=rng,
        )
        if result is not None:
            result.time_period = t
            all_matches.append(result)

    if not all_matches:
        return None

    return pl.DataFrame({
        tm: np.concatenate([[m.time_period] * len(m.treat_ids) for m in all_matches]),
        "treat_id": np.concatenate([m.treat_ids for m in all_matches]),
        "control_id": np.concatenate([m.control_ids for m in all_matches]),
        "difference": np.concatenate([m.differences for m in all_matches]),
    })
