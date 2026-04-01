"""
pyrollmatch — Fast rolling entry matching for staggered adoption studies.

A Python reimplementation of the R ``rollmatch`` package (RTI International)
using polars and numpy for scalable matching on large panel datasets (100K+ units).

Rolling entry matching (REM) explicitly handles staggered treatment adoption
by matching each treated unit to controls at the treated unit's specific entry
time, using accumulated (rolling-window) covariates.

Quick Start
-----------
>>> import polars as pl
>>> from pyrollmatch import rollmatch, alpha_sweep
>>>
>>> # data: panel with columns [unit_id, time, treat, entry_time, x1, x2, ...]
>>> result = rollmatch(
...     data, treat="treat", tm="time", entry="entry_time", id="unit_id",
...     covariates=["x1", "x2", "x3"],
...     alpha=0.1, num_matches=3,
... )
>>> result.balance  # SMD table
>>> result.weights  # unit_id -> matching weight

References
----------
- Witman et al. (2018). "Comparison Group Selection in the Presence of Rolling Entry."
  Health Services Research, 54(1), 262-270. doi:10.1111/1475-6773.13086
- RTI International rollmatch R package: https://github.com/RTIInternational/rollmatch
"""

from .core import rollmatch, alpha_sweep, RollmatchResult
from .reduce import reduce_data
from .score import score_data, ScoredResult
from .balance import compute_balance, smd_table
from .diagnostics import balance_test, equivalence_test

__version__ = "0.0.4"
__all__ = [
    "rollmatch",
    "alpha_sweep",
    "RollmatchResult",
    "reduce_data",
    "score_data",
    "ScoredResult",
    "compute_balance",
    "smd_table",
    "balance_test",
    "equivalence_test",
]
