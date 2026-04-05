# pyrollmatch API Reference

**Version 0.1.2**

## Overview

pyrollmatch implements rolling entry matching and weighting for staggered
treatment adoption studies. Built on polars (data) and numpy/scipy (computation)
for performance on large panel datasets.

**Pipeline:** `reduce_data()` → `score_data()` → matching/weighting → balance diagnostics

**Quick start:**
```python
from pyrollmatch import rollmatch

result = rollmatch(
    data, treat="treat", tm="time", entry="entry_time", id="unit_id",
    covariates=["x1", "x2", "x3"],
    ps_caliper=0.2, num_matches=1,
)
result.balance         # covariate balance table
result.matched_data    # match pairs
result.weights         # unit weights
```

---

## `rollmatch()`

Main entry point. Orchestrates reduce → score → match → weight → balance.

```python
rollmatch(
    data, treat, tm, entry, id, covariates,
    lookback=1, method="matching", verbose=True,
    **method_kwargs
) → RollmatchResult | None
```

### Core parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` | required | Panel data (unit × time) |
| `treat` | `str` | required | Binary treatment column (0/1) |
| `tm` | `str` | required | Time period column (integer) |
| `entry` | `str` | required | Entry period column |
| `id` | `str` | required | Unit identifier column |
| `covariates` | `list[str]` | required | Covariate column names |
| `lookback` | `int` | `1` | Periods before entry for baseline (≥ 1) |
| `method` | `str \| callable` | `"matching"` | `"matching"`, `"ebal"`, or callable |
| `verbose` | `bool` | `True` | Print progress |

### Matching kwargs (`method="matching"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ps_caliper` | `float` | `0` | PS caliper multiplier (0 = no caliper) |
| `ps_caliper_std` | `str` | `"average"` | Pooled SD method: `"average"`, `"weighted"`, `"none"` |
| `num_matches` | `int` | `1` | Controls per treated unit |
| `replacement` | `str` | `"cross_cohort"` | `"unrestricted"`, `"cross_cohort"`, `"global_no"` |
| `model_type` | `str` | `"logistic"` | Scoring model (see Model Types below) |
| `block_size` | `int` | `2000` | Block size for memory management |
| `mahvars` | `list[str] \| None` | `None` | Covariates for Mahalanobis matching with PS caliper |
| `m_order` | `str \| None` | `None` | Matching order: `"largest"`, `"smallest"`, `"random"`, `"data"` |
| `caliper` | `dict \| None` | `None` | Per-variable calipers: `{"x1": 0.5}` |
| `std_caliper` | `bool` | `True` | Per-variable caliper widths in SD units |

### Ebal kwargs (`method="ebal"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `moment` | `int` | `1` | Moment constraint (1=mean, 2=+variance, 3=+skewness) |
| `max_weight` | `float \| None` | `None` | Maximum weight cap |

---

## Model Types

### Propensity score models

Fit a classifier, produce scalar scores. Matching uses `|score_i - score_j|`.

| `model_type` | Description |
|--------------|-------------|
| `"logistic"` | Standard logistic regression (default) |
| `"probit"` | Probit model (inverse normal CDF) |
| `"gbm"` | Gradient boosting (`HistGradientBoostingClassifier`) |
| `"rf"` | Random forest |
| `"lasso"` | L1-regularized logistic |
| `"ridge"` | L2-regularized logistic |
| `"elasticnet"` | L1+L2 regularized logistic |

### Distance-based models

Compute pairwise distances in covariate space. No propensity model fitted.

| `model_type` | Description |
|--------------|-------------|
| `"mahalanobis"` | Pairwise Mahalanobis distance (pooled within-group covariance) |
| `"scaled_euclidean"` | Euclidean on covariates standardized by pooled within-group SD |
| `"robust_mahalanobis"` | Rank-based Mahalanobis (Rosenbaum 2010). Robust to outliers |
| `"euclidean"` | Raw Euclidean distance |

---

## Replacement Modes

Controls how control units are reused across matches.

| Mode | Within period | Across periods | Use case |
|------|--------------|----------------|----------|
| `"unrestricted"` | Reuse freely | Reuse freely | Maximum matches, PS matching |
| `"cross_cohort"` | No reuse | Reuse allowed | **Default.** Balanced within-period, flexible across |
| `"global_no"` | No reuse | No reuse | Strictest. Each control matched at most once globally |

---

## Matching Order (`m_order`)

Controls which treated units get matched first (matters when replacement is constrained).

| Value | Behavior | When to use |
|-------|----------|-------------|
| `"largest"` | Highest PS first | Default for PS matching. Hard-to-match units get first pick |
| `"smallest"` | Lowest PS first | |
| `"random"` | Random order | Reproducibility via `np.random.seed()` |
| `"data"` | Original data order | Default for distance models |
| `None` | Auto-detect | `"largest"` for PS models, `"data"` for distance models |

---

## The `mahvars` Pattern

Match on Mahalanobis distance of specific covariates while using a propensity
score caliper to restrict the matching pool. This is the MatchIt `mahvars`
pattern (Rubin 1980).

```python
result = rollmatch(
    data, ..., covariates=["x1", "x2", "x3"],
    ps_caliper=0.25,            # PS caliper on logistic scores
    mahvars=["x1", "x2"],       # match on Mahalanobis of x1, x2
)
```

- PS estimated on all `covariates` → used only for caliper
- Matching distance = Mahalanobis on `mahvars` covariates
- Cannot combine with distance-based `model_type`

---

## Per-Variable Calipers

Constrain individual covariates independently of the main matching distance.

```python
result = rollmatch(
    data, ...,
    caliper={"age": 0.5, "income": 0.3},  # max 0.5 SD for age, 0.3 SD for income
    std_caliper=True,                       # widths in SD units (default)
)
```

With `std_caliper=False`, widths are in raw covariate units.

---

## `RollmatchResult`

Returned by `rollmatch()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `matched_data` | `DataFrame \| None` | `[tm, treat_id, control_id, difference]`. `None` for ebal |
| `balance` | `DataFrame` | Covariate balance table with SMDs |
| `n_treated_total` | `int` | Total treated in reduced sample |
| `n_treated_matched` | `int` | Treated successfully matched/weighted |
| `n_controls_matched` | `int` | Unique controls used |
| `ps_caliper` | `float \| None` | PS caliper used. `None` for ebal/custom |
| `weights` | `DataFrame` | `[id, weight]` |
| `weighted_data` | `DataFrame \| None` | Per-cohort weights (ebal/custom only) |
| `method` | `str` | `"matching"`, `"ebal"`, or `"custom"` |

---

## `score_data()`

Scoring step. Always returns `ScoredResult`.

```python
score_data(
    reduced_data, covariates, treat,
    model_type="logistic", match_on="logit", max_iter=1000,
) → ScoredResult
```

### `ScoredResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `DataFrame` | Input + `"score"` column |
| `model` | `Any` | Fitted sklearn model (`None` for distance) |
| `covariates` | `list[str]` | Covariate names |
| `model_type` | `str` | Model type used |
| `match_on` | `str` | Score transformation |
| `cov_inv` | `ndarray \| None` | Inverse covariance (Mahalanobis) |
| `distance_metric` | `str \| None` | Distance identifier |
| `distance_transform` | `ndarray \| None` | Scaling matrix (scaled_euclidean) |
| `ranked_covariates` | `list[str] \| None` | Ranked column names (robust_mahalanobis) |

---

## `reduce_data()`

Construct quasi-panel for rolling entry matching.

```python
reduce_data(data, treat, tm, entry, id, lookback=1) → pl.DataFrame
```

Selects treated units at their baseline period (`entry - lookback`) and
controls at all matching time periods.

---

## Balance Diagnostics

```python
compute_balance(scored, matches, treat, id, tm, covariates) → DataFrame
compute_balance_weighted(data, weights, treat, id, covariates) → DataFrame
balance_by_period(scored, matches, treat, id, tm, covariates) → (agg, detail)
balance_by_period_weighted(data, weights, treat, id, tm, covariates) → (agg, detail)
smd_table(balance) → None  # prints formatted table
```

## Statistical Tests

```python
balance_test(scored, matches, treat, id, tm, covariates) → DataFrame
equivalence_test(scored, matches, treat, id, tm, covariates) → DataFrame
balance_test_weighted(data, weights, treat, id, covariates) → DataFrame
equivalence_test_weighted(data, weights, treat, id, covariates) → DataFrame
```

---

## `DistanceSpec`

Internal dataclass bundling distance parameters. Exported for advanced use.

```python
DistanceSpec(
    metric=None,       # "mahalanobis", "euclidean", etc.
    covariates=None,   # column names for distance computation
    cov_inv=None,      # inverse covariance matrix
    transform=None,    # scaling matrix
    is_mahvars=False,  # True if PS caliper should also apply
)
```

---

## Migration from v0.0.x

| Old (v0.0.x) | New (v0.1.0) |
|---------------|-------------|
| `alpha=0.2` | `ps_caliper=0.2` |
| `standard_deviation="average"` | `ps_caliper_std="average"` |
| `replacement=True` | `replacement="unrestricted"` |
| `replacement=False` | `replacement="cross_cohort"` |
| `match_on="logit"` (in rollmatch) | Removed from rollmatch. Use in `score_data()` directly |
| `score_data(..., return_model=True)` | `score_data(...)` (always returns `ScoredResult`) |
| `result.alpha` | `result.ps_caliper` |
| `num_matches` default 3 | Default is now 1 |
| `replacement` default `"unrestricted"` | Default is now `"cross_cohort"` |
| `lookback` max 10 | No upper bound (any integer ≥ 1) |
