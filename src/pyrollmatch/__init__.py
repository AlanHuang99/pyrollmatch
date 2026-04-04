"""
pyrollmatch — Rolling entry matching and weighting for staggered adoption studies.

A Python package for causal inference with staggered treatment adoption,
built on polars and numpy for scalable matching/weighting on large panel
datasets (100K+ units).

Methods
-------
- **Matching**: Nearest-neighbor matching on propensity scores or pairwise
  distances (Mahalanobis, Euclidean, robust Mahalanobis). Supports calipers,
  replacement modes, and matching order (following MatchIt conventions).
- **Entropy balancing**: Direct covariate balance via convex optimization
  (Hainmueller 2012). Each entry cohort weights independently (stacked design).
- **Custom**: User-defined per-period weighting functions.

Quick Start
-----------
>>> from pyrollmatch import rollmatch
>>>
>>> # Propensity score matching
>>> result = rollmatch(
...     data, treat="treat", tm="time", entry="entry_time", id="unit_id",
...     covariates=["x1", "x2", "x3"],
...     ps_caliper=0.2, num_matches=3,
... )
>>>
>>> # Mahalanobis distance matching
>>> result = rollmatch(
...     data, ..., model_type="mahalanobis",
... )
>>>
>>> # Entropy balancing
>>> result = rollmatch(
...     data, ..., method="ebal", moment=1,
... )
>>>
>>> result.balance         # covariate balance (SMD table)
>>> result.weights         # unit-level weights
>>> result.matched_data    # match pairs (matching only)

References
----------
- Witman et al. (2018). "Comparison Group Selection in the Presence of
  Rolling Entry." Health Services Research, 54(1), 262-270.
- Hainmueller, J. (2012). "Entropy Balancing for Causal Effects."
  Political Analysis, 20(1), 25-46.
- Imai, King, Stuart (2011). MatchIt: Nonparametric Preprocessing for
  Parametric Causal Inference.
- Rosenbaum, P. (2010). Design of Observational Studies, ch. 8.
"""

from .core import rollmatch, RollmatchResult
from .reduce import reduce_data
from .score import score_data, ScoredResult, SUPPORTED_MODELS, DISTANCE_MODELS, PROPENSITY_MODELS
from .match import DistanceSpec
from .weight import entropy_balance
from .balance import (
    compute_balance, compute_balance_weighted,
    balance_by_period, balance_by_period_weighted,
    smd_table,
)
from .diagnostics import (
    balance_test, equivalence_test,
    balance_test_weighted, equivalence_test_weighted,
)

__version__ = "0.1.2"
__all__ = [
    # Core
    "rollmatch",
    "RollmatchResult",
    # Pipeline stages
    "reduce_data",
    "score_data",
    "ScoredResult",
    # Distance
    "DistanceSpec",
    # Constants
    "SUPPORTED_MODELS",
    "DISTANCE_MODELS",
    "PROPENSITY_MODELS",
    # Weighting
    "entropy_balance",
    # Balance diagnostics
    "compute_balance",
    "compute_balance_weighted",
    "balance_by_period",
    "balance_by_period_weighted",
    "smd_table",
    # Statistical tests
    "balance_test",
    "equivalence_test",
    "balance_test_weighted",
    "equivalence_test_weighted",
]
