"""
score_data — Compute propensity scores or distance-based scores for matching.

Propensity score models estimate P(treat=1|X) and produce a scalar score
per unit. Distance-based models defer pairwise distance computation to the
matching engine and instead return covariate metadata (inverse covariance,
scaling transforms, ranked covariates) via :class:`ScoredResult`.

Supported model types
---------------------
Propensity score models:
    ``"logistic"``
        Standard logistic regression (default, same as R rollmatch).
    ``"probit"``
        Probit model via logistic regression + inverse normal CDF.
    ``"gbm"``
        Gradient boosting (``HistGradientBoostingClassifier``).
    ``"rf"``
        Random forest (``RandomForestClassifier``).
    ``"lasso"``
        L1-regularized logistic regression.
    ``"ridge"``
        L2-regularized logistic regression.
    ``"elasticnet"``
        L1+L2 regularized logistic regression.

Distance-based models:
    ``"mahalanobis"``
        Pairwise Mahalanobis distance using pooled within-group covariance
        (MatchIt convention: Rubin 1980).
    ``"scaled_euclidean"``
        Euclidean distance on covariates standardized by pooled within-group
        standard deviations.
    ``"robust_mahalanobis"``
        Rank-based Mahalanobis distance (Rosenbaum 2010, ch. 8). More robust
        to outliers than standard Mahalanobis.
    ``"euclidean"``
        Raw Euclidean distance between covariate vectors.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Any

from sklearn.linear_model import LogisticRegression
from scipy.special import ndtri
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Model type constants
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = (
    "logistic", "probit", "gbm", "rf",
    "lasso", "ridge", "elasticnet",
    "mahalanobis", "scaled_euclidean", "robust_mahalanobis", "euclidean",
)
"""All supported ``model_type`` values for :func:`score_data`."""

PROPENSITY_MODELS = ("logistic", "probit", "gbm", "rf", "lasso", "ridge", "elasticnet")
"""Model types that produce propensity scores (scalar per unit)."""

DISTANCE_MODELS = ("mahalanobis", "scaled_euclidean", "robust_mahalanobis", "euclidean")
"""Model types that compute pairwise distances in the matching engine."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScoredResult:
    """Container for scored data and distance/model metadata.

    Always returned by :func:`score_data`. Provides a uniform interface
    regardless of whether the model is propensity-based or distance-based.

    Attributes
    ----------
    data : pl.DataFrame
        Input data with an added ``"score"`` column. For propensity models,
        this contains real propensity scores. For distance models, it
        contains placeholder zeros (the matcher uses covariates directly).
    model : Any
        Fitted sklearn classifier for propensity models. ``None`` for
        distance-based models.
    covariates : list[str]
        Covariate column names used for scoring.
    model_type : str
        The model type used (e.g. ``"logistic"``, ``"mahalanobis"``).
    match_on : str
        Score transformation applied: ``"logit"`` or ``"pscore"``.
        Only meaningful for propensity models.
    cov_inv : np.ndarray or None
        Inverse covariance matrix. Set for ``"mahalanobis"`` and
        ``"robust_mahalanobis"``; ``None`` otherwise.
    distance_metric : str or None
        Distance metric identifier passed to the matcher:
        ``"mahalanobis"``, ``"robust_mahalanobis"``, ``"scaled_euclidean"``,
        ``"euclidean"``, or ``None`` for propensity score matching.
    distance_transform : np.ndarray or None
        Diagonal scaling matrix for ``"scaled_euclidean"``.
        ``None`` for all other model types.
    ranked_covariates : list[str] or None
        Column names of pre-ranked covariates added to ``data`` for
        ``"robust_mahalanobis"`` (e.g. ``["_ranked_x1", "_ranked_x2"]``).
        The matcher uses these instead of raw covariates so that
        full-dataset ranks are consistent with ``cov_inv``.
    """
    data: pl.DataFrame
    model: Any
    covariates: list[str]
    model_type: str
    match_on: str
    cov_inv: np.ndarray | None = None
    distance_metric: str | None = None
    distance_transform: np.ndarray | None = None
    ranked_covariates: list[str] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pooled_within_group_cov(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pooled within-group covariance matrix.

    Following MatchIt (``aux_functions.R`` ``pooled_cov``):
    group-mean-center each covariate, compute ``cov()``, then apply
    the degrees-of-freedom correction ``(n-1) / (n - n_groups)`` to
    get the unbiased pooled within-group estimate.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Covariate matrix.
    y : np.ndarray, shape (n,)
        Group labels (typically 0/1 for control/treated).

    Returns
    -------
    np.ndarray, shape (k, k)
        Pooled within-group covariance matrix.
    """
    n = len(y)
    unique_groups = np.unique(y)
    n_groups = len(unique_groups)
    # Vectorized group-mean centering
    group_means = np.zeros((n_groups, X.shape[1]), dtype=X.dtype)
    for i, g in enumerate(unique_groups):
        group_means[i] = X[y == g].mean(axis=0)
    group_idx = np.searchsorted(unique_groups, y)
    X_centered = X - group_means[group_idx]
    # R's cov() divides by (n-1); MatchIt multiplies by (n-1)/(n-n_groups)
    cov = np.cov(X_centered, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    cov *= (n - 1) / (n - n_groups)
    return cov


def _pooled_within_group_sd(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pooled within-group standard deviations per covariate.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
    y : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray, shape (k,)
    """
    cov = _pooled_within_group_cov(X, y)
    return np.sqrt(np.diag(cov))


def _build_model(model_type: str, max_iter: int = 1000, random_state: int = 42):
    """Build an unfitted sklearn classifier for propensity score estimation."""
    if model_type in ("logistic", "probit"):
        return LogisticRegression(
            max_iter=max_iter, solver="lbfgs", random_state=random_state,
        )
    elif model_type == "lasso":
        return LogisticRegression(
            penalty="l1", solver="saga", max_iter=max_iter,
            random_state=random_state,
        )
    elif model_type == "ridge":
        return LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=max_iter,
            random_state=random_state,
        )
    elif model_type == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, solver="saga",
            max_iter=max_iter, random_state=random_state,
        )
    elif model_type == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, learning_rate=0.1,
            random_state=random_state,
        )
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _predict_scores(model, X: np.ndarray, model_type: str, match_on: str) -> np.ndarray:
    """Predict propensity scores from a fitted model.

    Parameters
    ----------
    model : fitted sklearn classifier
    X : np.ndarray, shape (n, k)
    model_type : str
    match_on : {"logit", "pscore"}
        ``"logit"``: log-odds (or probit z-score). ``"pscore"``: raw P(T=1|X).

    Returns
    -------
    np.ndarray, shape (n,)
    """
    proba = model.predict_proba(X)[:, 1]
    proba_clipped = np.clip(proba, 1e-10, 1 - 1e-10)

    if match_on == "pscore":
        return proba
    elif match_on == "logit":
        if model_type == "probit":
            return ndtri(proba_clipped)
        else:
            return np.log(proba_clipped / (1 - proba_clipped))
    else:
        raise ValueError(f"match_on must be 'logit' or 'pscore', got '{match_on}'")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_data(
    reduced_data: pl.DataFrame,
    covariates: list[str],
    treat: str,
    model_type: str = "logistic",
    match_on: str = "logit",
    max_iter: int = 1000,
) -> ScoredResult:
    """Fit a propensity/distance model and return scored data with metadata.

    This is the scoring step of the matching pipeline. For propensity score
    models, a classifier is fitted and scores are added to the data. For
    distance-based models, covariance / scaling metadata is computed and
    returned in the :class:`ScoredResult` for the matching engine to use.

    Parameters
    ----------
    reduced_data : pl.DataFrame
        Output from :func:`reduce_data`.
    covariates : list[str]
        Column names of matching covariates.
    treat : str
        Column name for binary treatment indicator (1=treated, 0=control).
    model_type : str, default ``"logistic"``
        Scoring model. One of :data:`SUPPORTED_MODELS`:

        **Propensity score models** (produce scalar scores per unit):
        ``"logistic"``, ``"probit"``, ``"gbm"``, ``"rf"``,
        ``"lasso"``, ``"ridge"``, ``"elasticnet"``.

        **Distance-based models** (pairwise distances computed by matcher):
        ``"mahalanobis"``, ``"scaled_euclidean"``,
        ``"robust_mahalanobis"``, ``"euclidean"``.
    match_on : str, default ``"logit"``
        Score transformation for propensity models:
        ``"logit"`` for log-odds (recommended), ``"pscore"`` for raw
        probability. Ignored for distance-based models.
    max_iter : int, default 1000
        Maximum optimizer iterations (propensity models only).

    Returns
    -------
    ScoredResult
        Contains ``.data`` (DataFrame with ``"score"`` column),
        ``.model`` (fitted classifier or None), and distance metadata.

    Raises
    ------
    ValueError
        If ``model_type`` is not in :data:`SUPPORTED_MODELS`, covariates
        are missing, or data contains NaN values.

    Examples
    --------
    >>> result = score_data(reduced, ["x1", "x2"], "treat")
    >>> result.data["score"]        # propensity scores
    >>> result.model                # fitted LogisticRegression

    >>> result = score_data(reduced, ["x1", "x2"], "treat",
    ...                     model_type="mahalanobis")
    >>> result.cov_inv              # inverse covariance matrix
    >>> result.distance_metric      # "mahalanobis"
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'"
        )

    valid_match = ("logit", "pscore")
    if match_on not in valid_match and model_type not in DISTANCE_MODELS:
        raise ValueError(f"match_on must be one of {valid_match}, got '{match_on}'")

    for col in covariates:
        if col not in reduced_data.columns:
            raise ValueError(f"Covariate '{col}' not found in data")

    # Extract numpy arrays
    X = reduced_data.select(covariates).to_numpy().astype(np.float64)
    y = reduced_data[treat].to_numpy().astype(np.int32)

    # Check for NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        raise ValueError(
            f"{nan_mask.sum()} rows have NaN in covariates. "
            "Remove NaN rows before scoring."
        )

    # Initialize metadata fields
    cov_inv = None
    distance_metric = None
    distance_transform = None
    ranked_covariates = None

    if model_type in DISTANCE_MODELS:
        scores = np.zeros(len(X))  # placeholder — matcher uses covariates
        model = None

        if model_type == "mahalanobis":
            cov = _pooled_within_group_cov(X, y)
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov)
            distance_metric = "mahalanobis"

        elif model_type == "scaled_euclidean":
            sd = _pooled_within_group_sd(X, y)
            sd[sd < 1e-10] = 1.0
            distance_transform = np.diag(1.0 / sd)
            distance_metric = "scaled_euclidean"

        elif model_type == "robust_mahalanobis":
            # MatchIt: rank full sample, cov(ranks), scale by sd(1:n)
            n = X.shape[0]
            X_ranked = np.column_stack([
                rankdata(X[:, j], method="average") for j in range(X.shape[1])
            ])
            var_r = np.cov(X_ranked, rowvar=False, ddof=1)
            var_r = np.atleast_2d(var_r)
            sd_1_to_n = np.std(np.arange(1, n + 1), ddof=1)
            multiplier = sd_1_to_n / np.sqrt(np.diag(var_r)).clip(1e-10)
            var_r = var_r * np.outer(multiplier, multiplier)
            var_r += np.eye(var_r.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(var_r)
            distance_metric = "robust_mahalanobis"

        elif model_type == "euclidean":
            distance_metric = "euclidean"

    else:
        # Propensity score model
        model = _build_model(model_type, max_iter)
        model.fit(X, y)
        scores = _predict_scores(model, X, model_type, match_on)

    result_df = reduced_data.with_columns(pl.Series("score", scores))

    # For robust_mahalanobis: add full-dataset ranked columns
    if model_type == "robust_mahalanobis":
        ranked_covariates = [f"_ranked_{c}" for c in covariates]
        result_df = result_df.with_columns([
            pl.Series(f"_ranked_{col}", X_ranked[:, j])
            for j, col in enumerate(covariates)
        ])

    return ScoredResult(
        data=result_df, model=model, covariates=covariates,
        model_type=model_type, match_on=match_on,
        cov_inv=cov_inv,
        distance_metric=distance_metric,
        distance_transform=distance_transform,
        ranked_covariates=ranked_covariates,
    )
