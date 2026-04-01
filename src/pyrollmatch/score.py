"""
score_data — Compute propensity scores or distance-based scores for matching.

Supports multiple model types:
  - logistic: Logistic regression (default, same as R rollmatch)
  - probit: Probit model (logistic + inverse normal CDF transform)
  - gbm: Gradient boosting (HistGradientBoostingClassifier, fast + non-linear)
  - rf: Random forest (RandomForestClassifier)
  - lasso: L1-regularized logistic regression
  - ridge: L2-regularized logistic regression
  - elasticnet: L1+L2 regularized logistic regression
  - mahalanobis: No propensity model — uses Mahalanobis distance directly

All propensity models output either logit-transformed scores or raw
probabilities, which are then used by the matching engine.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import Any

from sklearn.linear_model import LogisticRegression
from scipy.special import ndtri
from scipy.spatial.distance import cdist


# All supported model types
SUPPORTED_MODELS = (
    "logistic", "probit", "gbm", "rf",
    "lasso", "ridge", "elasticnet", "mahalanobis",
)

# Models that produce propensity scores (vs distance-based)
PROPENSITY_MODELS = ("logistic", "probit", "gbm", "rf", "lasso", "ridge", "elasticnet")


@dataclass
class ScoredResult:
    """Container for scored data and the fitted model."""
    data: pl.DataFrame
    model: Any  # sklearn classifier or None for mahalanobis
    covariates: list[str]
    model_type: str
    match_on: str


def _build_model(model_type: str, max_iter: int = 1000, random_state: int = 42):
    """Build the appropriate sklearn classifier.

    Parameters
    ----------
    model_type : str
        One of the PROPENSITY_MODELS.
    max_iter : int
        Max iterations for iterative solvers.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    sklearn classifier instance (unfitted).
    """
    if model_type in ("logistic", "probit"):
        return LogisticRegression(
            max_iter=max_iter, solver="lbfgs", random_state=random_state,
        )
    elif model_type == "lasso":
        return LogisticRegression(
            l1_ratio=1.0, solver="saga", max_iter=max_iter,
            random_state=random_state,
        )
    elif model_type == "ridge":
        return LogisticRegression(
            l1_ratio=0.0, solver="lbfgs", max_iter=max_iter,
            random_state=random_state,
        )
    elif model_type == "elasticnet":
        return LogisticRegression(
            l1_ratio=0.5, solver="saga",
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
    """Predict propensity scores from fitted model.

    Parameters
    ----------
    model : fitted sklearn classifier
    X : numpy array of covariates
    model_type : str
    match_on : "logit" or "pscore"

    Returns
    -------
    numpy array of scores
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


def score_data(
    reduced_data: pl.DataFrame,
    covariates: list[str],
    treat: str,
    model_type: str = "logistic",
    match_on: str = "logit",
    max_iter: int = 1000,
    return_model: bool = False,
) -> pl.DataFrame | ScoredResult:
    """Fit propensity model and add scores to data.

    Parameters
    ----------
    reduced_data : pl.DataFrame
        Output from reduce_data().
    covariates : list[str]
        Column names of matching covariates.
    treat : str
        Column name for binary treatment indicator.
    model_type : str
        Model type for propensity score estimation:
        - "logistic": Standard logistic regression (default, matches R rollmatch)
        - "probit": Probit model (inverse normal CDF transform)
        - "gbm": Gradient boosting (HistGradientBoostingClassifier — fast, non-linear)
        - "rf": Random forest (captures interactions automatically)
        - "lasso": L1-regularized logistic (variable selection)
        - "ridge": L2-regularized logistic (handles multicollinearity)
        - "elasticnet": L1+L2 regularized logistic
        - "mahalanobis": No propensity model — scores are Mahalanobis distances
          from each unit to the treated group centroid. Matching uses covariate
          space directly.
    match_on : str
        "logit" for log-odds/probit-transformed score (default), "pscore" for
        raw probability. Ignored for mahalanobis.
    max_iter : int
        Maximum iterations for the optimizer (propensity models only).
    return_model : bool
        If True, return ScoredResult with fitted model. Otherwise return DataFrame.

    Returns
    -------
    pl.DataFrame (default)
        Input data with added "score" column.
    ScoredResult (if return_model=True)
        Contains .data, .model, .covariates, .model_type, .match_on.
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'"
        )

    valid_match = ("logit", "pscore")
    if match_on not in valid_match and model_type != "mahalanobis":
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

    if model_type == "mahalanobis":
        # No propensity model — compute Mahalanobis distance to treated centroid
        # This gives each unit a "score" = distance from treated group center
        # Matching then pairs units with similar distances
        X_treated = X[y == 1]
        centroid = X_treated.mean(axis=0)

        # Covariance from pooled sample (more stable than treated-only)
        cov = np.cov(X, rowvar=False)
        # Regularize to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6

        cov_inv = np.linalg.inv(cov)
        diff = X - centroid
        scores = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        model = None
    else:
        # Propensity score model
        model = _build_model(model_type, max_iter)
        model.fit(X, y)
        scores = _predict_scores(model, X, model_type, match_on)

    result_df = reduced_data.with_columns(pl.Series("score", scores))

    if return_model:
        return ScoredResult(
            data=result_df, model=model, covariates=covariates,
            model_type=model_type, match_on=match_on,
        )

    return result_df
