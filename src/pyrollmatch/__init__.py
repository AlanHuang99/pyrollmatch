"""
pyrollmatch — Rolling entry matching and weighting for staggered adoption studies.

A Python package for causal inference with staggered treatment adoption,
using polars and numpy for scalable matching/weighting on large panel
datasets (100K+ units).

Supports pluggable methods: propensity score matching, entropy balancing
(Hainmueller 2012), and user-defined weighting functions through a
unified per-period interface.

Quick Start
-----------
>>> import polars as pl
>>> from pyrollmatch import rollmatch
>>>
>>> # Propensity score matching (default)
>>> result = rollmatch(
...     data, treat="treat", tm="time", entry="entry_time", id="unit_id",
...     covariates=["x1", "x2", "x3"],
...     method="matching", alpha=0.1, num_matches=3,
... )
>>>
>>> # Entropy balancing
>>> result = rollmatch(
...     data, treat="treat", tm="time", entry="entry_time", id="unit_id",
...     covariates=["x1", "x2", "x3"],
...     method="ebal", moment=1,
... )
>>>
>>> result.balance   # SMD table
>>> result.weights   # unit_id -> weight

References
----------
- Witman et al. (2018). "Comparison Group Selection in the Presence of Rolling Entry."
  Health Services Research, 54(1), 262-270. doi:10.1111/1475-6773.13086
- Hainmueller, J. (2012). "Entropy Balancing for Causal Effects."
  Political Analysis, 20(1), 25-46. doi:10.1093/pan/mpr025
- RTI International rollmatch R package: https://github.com/RTIInternational/rollmatch
"""

from .core import rollmatch, RollmatchResult
from .reduce import reduce_data
from .score import score_data, ScoredResult
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

__version__ = "0.0.8"
__all__ = [
    "rollmatch",
    "RollmatchResult",
    "reduce_data",
    "score_data",
    "ScoredResult",
    "entropy_balance",
    "compute_balance",
    "compute_balance_weighted",
    "balance_by_period",
    "balance_by_period_weighted",
    "smd_table",
    "balance_test",
    "equivalence_test",
    "balance_test_weighted",
    "equivalence_test_weighted",
]
