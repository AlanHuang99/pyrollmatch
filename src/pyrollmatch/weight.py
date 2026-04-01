"""
weight — Weight computation for matching and balancing methods.

Includes:
  - _compute_weights: derive IPW weights from match pairs
  - entropy_balance: entropy balancing (Hainmueller 2012) for direct
    covariate balance without propensity scores
"""

import warnings
import numpy as np
import polars as pl
from scipy.optimize import minimize


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


# ---------------------------------------------------------------------------
# Entropy balancing (Hainmueller 2012)
# ---------------------------------------------------------------------------

def _build_constraint_matrix(X: np.ndarray, moment: int) -> np.ndarray:
    """Build the constraint matrix for entropy balancing.

    Parameters
    ----------
    X : array of shape (n, k)
        Control covariate matrix.
    moment : int (1, 2, or 3)
        Which moments to balance.

    Returns
    -------
    array of shape (n, d) where d depends on moment:
        moment=1: [1, X]              (k+1 columns)
        moment=2: [1, X, X²]          (2k+1 columns)
        moment=3: [1, X, X², X³]      (3k+1 columns)
    """
    n = X.shape[0]
    parts = [np.ones((n, 1)), X]
    if moment >= 2:
        parts.append(X ** 2)
    if moment >= 3:
        parts.append(X ** 3)
    return np.hstack(parts)


def _build_target(X_t: np.ndarray, moment: int) -> np.ndarray:
    """Build the target vector (treated moments to match).

    Parameters
    ----------
    X_t : array of shape (n_t, k)
        Treated covariate matrix.
    moment : int

    Returns
    -------
    array of shape (d,) matching the constraint matrix columns.
    """
    parts = [np.array([1.0])]  # normalization: weights sum to 1
    parts.append(np.mean(X_t, axis=0))
    if moment >= 2:
        parts.append(np.mean(X_t ** 2, axis=0))
    if moment >= 3:
        parts.append(np.mean(X_t ** 3, axis=0))
    return np.concatenate(parts)


def _dual_objective(lam, C, target, base_q):
    """Evaluate the entropy balancing dual objective and gradient.

    The dual problem (Hainmueller 2012, Appendix) is:

        min_λ  L(λ) = Σ_i q_i · exp(C_i · λ) − λ · target

    where C is the constraint matrix, target is the treated moment vector,
    and q_i are base weights (uniform: 1/N_c).

    Parameters
    ----------
    lam : array of shape (d,)
        Lagrange multipliers.
    C : array of shape (n, d)
        Constraint matrix (standardized).
    target : array of shape (d,)
        Target moments (standardized).
    base_q : array of shape (n,)
        Base weights.

    Returns
    -------
    (objective, gradient) : (float, array of shape (d,))
    """
    # exp(C @ λ) with overflow protection
    lin = C @ lam
    lin = np.clip(lin, -500, 500)
    exp_lin = np.exp(lin)

    w = base_q * exp_lin  # shape (n,)

    obj = float(np.sum(w) - lam @ target)
    grad = C.T @ w - target  # shape (d,)

    return obj, grad


def entropy_balance(
    treated_data: pl.DataFrame,
    control_data: pl.DataFrame,
    covariates: list[str],
    id: str,
    moment: int = 1,
    max_weight: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> pl.DataFrame | None:
    """Compute entropy-balanced weights for control units.

    Finds weights that minimize KL divergence from uniform while
    satisfying exact covariate moment constraints (Hainmueller 2012).

    Parameters
    ----------
    treated_data : pl.DataFrame
        Treated units for this cohort.
    control_data : pl.DataFrame
        Control pool for this cohort.
    covariates : list[str]
        Covariate columns to balance.
    id : str
        Unit identifier column.
    moment : int (1, 2, or 3)
        Moments to balance: 1=means, 2=means+variances, 3=+skewness.
    max_weight : float or None
        Optional upper bound on any single weight (after normalization).
    tol : float
        Convergence tolerance for the optimizer.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    pl.DataFrame with columns [id, weight] for treated (weight=1) and
    controls (entropy-balanced weights), or None if optimization fails.
    """
    X_t = treated_data.select(covariates).to_numpy().astype(np.float64)
    X_c = control_data.select(covariates).to_numpy().astype(np.float64)

    n_t = X_t.shape[0]
    n_c = X_c.shape[0]

    if n_t == 0 or n_c == 0:
        return None

    # Build constraint matrix and target
    C = _build_constraint_matrix(X_c, moment)
    target = _build_target(X_t, moment)

    # Standardize for numerical stability
    col_mean = C.mean(axis=0)
    col_std = C.std(axis=0)
    col_std[col_std < 1e-10] = 1.0  # avoid division by zero for constant cols
    # Don't standardize the intercept column
    col_mean[0] = 0.0
    col_std[0] = 1.0

    C_std = (C - col_mean) / col_std
    target_std = (target - col_mean) / col_std

    # Base weights (uniform)
    base_q = np.full(n_c, 1.0 / n_c)

    # Solve the dual
    lam0 = np.zeros(C_std.shape[1])
    result = minimize(
        _dual_objective,
        lam0,
        args=(C_std, target_std, base_q),
        method="L-BFGS-B",
        jac=True,  # _dual_objective returns (obj, grad)
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    if not result.success:
        warnings.warn(
            f"Entropy balancing did not converge: {result.message}. "
            "This may indicate insufficient overlap between treated and "
            "control covariate distributions.",
            stacklevel=2,
        )
        return None

    # Recover primal weights
    lin = C_std @ result.x
    lin = np.clip(lin, -500, 500)
    w = base_q * np.exp(lin)

    # w should sum to ~1 (enforced by intercept constraint); rescale to n_treated
    w = w * (n_t / w.sum())

    # Effective sample size check
    n_eff = w.sum() ** 2 / np.sum(w ** 2)
    if n_eff / n_c < 0.1:
        warnings.warn(
            f"Effective sample size is very low: n_eff={n_eff:.1f} "
            f"({100*n_eff/n_c:.1f}% of {n_c} controls). "
            "Consider relaxing moment constraints or checking overlap.",
            stacklevel=2,
        )

    # Optional weight capping (iterate to handle re-normalization)
    if max_weight is not None:
        for _ in range(20):  # converges in a few iterations
            if np.max(w) <= max_weight:
                break
            w = np.minimum(w, max_weight)
            w = w * (n_t / w.sum())

    # Build output DataFrame
    treat_ids = treated_data[id].to_numpy()
    ctrl_ids = control_data[id].to_numpy()

    treat_weights = pl.DataFrame({
        id: treat_ids,
        "weight": np.ones(n_t),
    })
    ctrl_weights = pl.DataFrame({
        id: ctrl_ids,
        "weight": w,
    })

    return pl.concat([treat_weights, ctrl_weights])
