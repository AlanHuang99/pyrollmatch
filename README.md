# pyrollmatch

Matching and weighting for staggered treatment adoption studies in Python, built on polars and numpy.

## Installation

```bash
pip install pyrollmatch
```

**From source:**
```bash
git clone https://github.com/AlanHuang99/pyrollmatch.git
cd pyrollmatch
pip install -e ".[dev]"
```

## Quick Start

```python
import polars as pl
from pyrollmatch import rollmatch

data = pl.read_parquet("panel_data.parquet")

result = rollmatch(
    data,
    treat="treat",
    tm="time",
    entry="entry_time",
    id="unit_id",
    covariates=["x1", "x2", "x3"],
    ps_caliper=0.2,
    num_matches=1,
)

result.balance         # covariate balance table (SMDs)
result.matched_data    # match pairs: [tm, treat_id, control_id, difference]
result.weights         # unit-level weights: [id, weight]
```

## Methods

### Nearest-Neighbor Matching (default)

Greedy nearest-neighbor matching on propensity scores or pairwise covariate distances. Supports PS calipers, per-variable calipers, matching order, and three replacement modes.

```python
# Propensity score matching (logistic regression)
result = rollmatch(data, ..., ps_caliper=0.2, num_matches=1)

# Mahalanobis distance matching
result = rollmatch(data, ..., model_type="mahalanobis")

# Mahalanobis matching with PS caliper (MatchIt mahvars pattern)
result = rollmatch(data, ..., ps_caliper=0.25, mahvars=["x1", "x2"])
```

### Entropy Balancing

Direct covariate balance via convex optimization (Hainmueller 2012). Each entry cohort weights the full control pool independently (stacked design).

```python
result = rollmatch(data, ..., method="ebal", moment=1)

result.weights         # unit weights
result.weighted_data   # per-cohort weights [tm, id, weight]
```

### Custom Methods

User-defined per-period weighting functions:

```python
def my_method(treated_data, control_data, covariates, id, **kwargs):
    # Return pl.DataFrame with columns [id, weight]
    ...

result = rollmatch(data, ..., method=my_method)
```

---

## Scoring Models

11 model types via the `model_type` parameter:

### Propensity score models

Fit a classifier, match on `|score_i - score_j|`.

| `model_type` | Description |
|---|---|
| `"logistic"` | Standard logistic regression **(default)** |
| `"probit"` | Probit model (inverse normal CDF) |
| `"gbm"` | Gradient boosting (`HistGradientBoostingClassifier`) |
| `"rf"` | Random forest |
| `"lasso"` | L1-regularized logistic |
| `"ridge"` | L2-regularized logistic |
| `"elasticnet"` | L1+L2 regularized logistic |

### Distance-based models

Pairwise covariate distances. No propensity model fitted.

| `model_type` | Description |
|---|---|
| `"mahalanobis"` | Mahalanobis distance (pooled within-group covariance, MatchIt convention) |
| `"scaled_euclidean"` | Euclidean on covariates standardized by pooled within-group SD |
| `"robust_mahalanobis"` | Rank-based Mahalanobis (Rosenbaum 2010). Robust to outliers |
| `"euclidean"` | Raw Euclidean distance |

---

## Matching Parameters

### Caliper

```python
# Propensity score caliper: ps_caliper * pooled_SD
result = rollmatch(data, ..., ps_caliper=0.2)

# Per-variable calipers (in SD units by default)
result = rollmatch(data, ..., caliper={"age": 0.5, "income": 0.3})

# Per-variable calipers in raw units
result = rollmatch(data, ..., caliper={"age": 5}, std_caliper=False)
```

`ps_caliper_std` controls how the pooled SD is computed: `"average"` (default), `"weighted"`, or `"none"` (raw units).

### Replacement Modes

Controls whether control units can be reused across matches.

| `replacement=` | Within period | Across periods | Use case |
|---|---|---|---|
| `"unrestricted"` | Reuse freely | Reuse freely | Maximize match rate |
| **`"cross_cohort"`** | No reuse | Reuse allowed | **Default.** Balanced within-period |
| `"global_no"` | No reuse | No reuse | Strictest. Each control used at most once |

### Matching Order (`m_order`)

Controls which treated units are matched first. Matters when replacement is constrained.

| `m_order=` | Behavior |
|---|---|
| `"largest"` | Highest PS first **(default for PS models)**. Hard-to-match units get first pick |
| `"smallest"` | Lowest PS first |
| `"random"` | Random order |
| `"data"` | Original data order **(default for distance models)** |

### The `mahvars` Pattern

Match on Mahalanobis distance of specific covariates while using a propensity score caliper to restrict the pool. Follows the MatchIt `mahvars` convention.

```python
result = rollmatch(
    data, ...,
    covariates=["x1", "x2", "x3"],   # PS estimated on all covariates
    ps_caliper=0.25,                   # PS caliper for pool restriction
    mahvars=["x1", "x2"],             # Mahalanobis matching on these
)
```

---

## Balance Diagnostics

### Post-Matching Balance

```python
from pyrollmatch import balance_test, equivalence_test

# SMD + t-test + variance ratio + KS test
diag = balance_test(scored_data, result.matched_data,
                    "treat", "unit_id", "time", covariates)

# TOST equivalence test (Hartman & Hidalgo 2018)
equiv = equivalence_test(scored_data, result.matched_data,
                         "treat", "unit_id", "time", covariates)
```

### Per-Period Balance

Pooled SMD can mask within-cohort imbalance. Check each entry cohort:

```python
from pyrollmatch import balance_by_period

agg, detail = balance_by_period(
    scored_data, result.matched_data,
    "treat", "unit_id", "time", covariates,
)
# agg: covariate × {wtd_mean_smd, median_abs_smd, max_abs_smd}
# detail: period × covariate × {n_treated, n_controls, smd}
```

### Weighted Diagnostics (for ebal/custom)

```python
from pyrollmatch import balance_test_weighted, equivalence_test_weighted

diag = balance_test_weighted(reduced_data, result.weights,
                             "treat", "unit_id", covariates)
```

---

## Data Format

Input: `polars.DataFrame` in **long panel format** (one row per unit per time period).

| Column | Type | Description |
|---|---|---|
| `id` | int/str | Unit identifier |
| `tm` | int | Time period (integer, monotonically increasing) |
| `treat` | int (0/1) | **Time-invariant** treatment group indicator. 1 = eventually treated, 0 = never treated |
| `entry` | int | Treatment onset period for treated units. For controls: any value > max(tm) or null |
| covariates | float | Matching variables |

```
unit_id | time | treat | entry_time | x1   | x2
--------|------|-------|------------|------|-----
1       | 1    | 1     | 5          | 2.3  | 1.1   <- treated, enters period 5
1       | 2    | 1     | 5          | 2.5  | 1.0
...
101     | 1    | 0     | 99         | 1.8  | 0.9   <- control (entry=99 sentinel)
101     | 2    | 0     | 99         | 1.9  | 1.0
```

---

## Pipeline

For advanced use, each step is independently callable:

```python
from pyrollmatch import reduce_data, score_data, compute_balance

# 1. Reduce: select baseline covariates for each entry cohort
reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id", lookback=1)

# 2. Score: fit model, compute scores/distances
scored = score_data(reduced, ["x1", "x2", "x3"], "treat", model_type="logistic")
scored.data          # DataFrame with "score" column
scored.model         # fitted sklearn classifier

# 3. Match: via rollmatch() or match_all_periods() directly

# 4. Balance: assess covariate balance
balance = compute_balance(scored.data, matches, "treat", "unit_id", "time", covariates)
```

---

## API Reference

- **[REFERENCE.md](REFERENCE.md)** — parameter tables, return types, migration guide
- **[API docs](https://alanhuang99.github.io/pyrollmatch/)** — searchable HTML reference (auto-generated from docstrings)

---

## Testing

```bash
uv run pytest tests/           # 151 tests
uv run pytest tests/ -k stress # stress/scale tests
```

Tests include synthetic data, the Lalonde dataset, and a staggered panel fixture.

---

## Acknowledgements

Inspired by the [rollmatch](https://github.com/RTIInternational/rollmatch) R package by RTI International (Witman et al. 2018). Distance metrics and matching conventions follow [MatchIt](https://kosukeimai.github.io/MatchIt/) (Imai, King, Stuart 2011).

## References

- Witman, A., et al. (2018). "Comparison Group Selection in the Presence of Rolling Entry." *Health Services Research*, 54(1), 262-270.
- Hainmueller, J. (2012). "Entropy Balancing for Causal Effects." *Political Analysis*, 20(1), 25-46.
- Imai, K., King, G., Stuart, E. (2011). MatchIt: Nonparametric Preprocessing for Parametric Causal Inference. *Journal of Statistical Software*, 42(8).
- Rosenbaum, P. (2010). *Design of Observational Studies*, ch. 8.
- Hartman, E. & Hidalgo, F. D. (2018). "An Equivalence Approach to Balance and Placebo Tests." *American Journal of Political Science*, 62(4), 1000-1013.

## License

MIT
