# pyrollmatch

> **Alpha** (v0.1.x) — This package is in early development. Results have been validated against R rollmatch on synthetic data, but edge cases may remain. APIs are not stable and may change without notice. **Please verify critical results independently.**

Fast rolling entry matching for staggered adoption studies in Python.

High-performance reimplementation of the R [rollmatch](https://github.com/RTIInternational/rollmatch) package using **polars** + **numpy**, with 8 propensity score models and comprehensive post-matching diagnostics.

| | R rollmatch | pyrollmatch |
|---|---|---|
| Max scale | ~5K treated (crashes) | **90K+ treated** |
| 10K × 30K | OOM (5.8B row join) | **1.7 seconds** |
| Scoring models | Logistic only | **8 models** |
| Diagnostics | SMD only | SMD + t-test + VR + KS + TOST |
| Data library | dplyr (deprecated APIs) | polars |

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
from pyrollmatch import rollmatch, alpha_sweep

# Panel data: unit × time with treatment indicator
data = pl.read_parquet("panel_data.parquet")

# Rolling entry matching
result = rollmatch(
    data,
    treat="treat",              # binary treatment group indicator (1=treated, 0=control)
    tm="time_period",           # time period (integer)
    entry="entry_time",         # treatment onset period for treated units;
                                # for controls, set to any value > max(time_period)
    id="unit_id",               # unit identifier
    covariates=["x1", "x2"],    # matching covariates
    lookback=1,                 # periods to look back for baseline
    alpha=0.1,                  # caliper = alpha × pooled_SD
    num_matches=3,              # controls per treated
    replacement="cross_cohort", # control reuse policy (see below)
    model_type="logistic",      # propensity score model
)

# Results
result.balance      # SMD table (polars DataFrame)
result.weights      # unit_id → weight (polars DataFrame)
result.matched_data # treat_id, control_id, difference
```

### Alpha Sweep

Find the optimal caliper automatically:

```python
summary, best = alpha_sweep(
    data, treat="treat", tm="time", entry="entry_time", id="unit_id",
    covariates=["x1", "x2", "x3"],
    alphas=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
)
# summary: alpha, match_rate, max|SMD|, all_pass
# best: RollmatchResult from the best alpha
```

### Control Replacement Modes

The `replacement` parameter controls whether a control unit can be reused across matching:

| Mode | `replacement=` | Within-period reuse | Cross-period reuse | Best for |
|---|---|---|---|---|
| **Unrestricted** | `"unrestricted"` or `True` | Yes | Yes | Maximizing match rate |
| **Cross-cohort** | `"cross_cohort"` or `False` | No | Yes | R rollmatch compatibility |
| **Global no** | `"global_no"` | No | No | Unique control assignments |

```python
# Each control matched at most once across ALL periods
result = rollmatch(data, ..., replacement="global_no")

# No reuse within a period, but allowed across periods (R rollmatch default)
result = rollmatch(data, ..., replacement="cross_cohort")

# Controls can match multiple treated units freely
result = rollmatch(data, ..., replacement="unrestricted")
```

> **Note on `"global_no"`**: Because controls are consumed as periods are processed (earliest first), the order of periods affects results. This mode also changes the estimand — later cohorts may get worse matches as the control pool shrinks. Use `balance_by_period()` to check whether later cohorts suffer.

> **Backward compatibility**: `replacement=True` maps to `"unrestricted"` and `replacement=False` maps to `"cross_cohort"`, matching the original behavior.

### Per-Period Balance Diagnostics

Pooled SMD can mask within-cohort imbalance through cancellation. Use `balance_by_period()` to check each entry cohort individually:

```python
from pyrollmatch import balance_by_period, reduce_data, score_data

reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
scored = score_data(reduced, ["x1", "x2", "x3"], "treat")

agg, detail = balance_by_period(
    scored, result.matched_data,
    "treat", "unit_id", "time", ["x1", "x2", "x3"],
)

# agg: covariate × {wtd_mean_smd, median_abs_smd, max_abs_smd, n_periods}
# detail: period × covariate × {n_treated, n_controls, smd}
```

The aggregate table reports three summary measures per covariate:
- **`wtd_mean_smd`** — weighted mean SMD (weighted by n_treated per period)
- **`median_abs_smd`** — robust central tendency
- **`max_abs_smd`** — most conservative, catches the worst cohort

### Post-Matching Diagnostics

```python
from pyrollmatch import balance_test, equivalence_test
from pyrollmatch import reduce_data, score_data

reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id")
scored = score_data(reduced, ["x1", "x2", "x3"], "treat")

# SMD + t-test + variance ratio + KS test
diag = balance_test(scored, result.matched_data,
                    "treat", "unit_id", "time", ["x1", "x2", "x3"])

# TOST equivalence test (Hartman & Hidalgo 2018)
equiv = equivalence_test(scored, result.matched_data,
                         "treat", "unit_id", "time", ["x1", "x2", "x3"])
```

## Propensity Score Models

8 scoring methods, all accessible via the `model_type` parameter:

| Model | `model_type` | Description | Best for |
|---|---|---|---|
| **Logistic** | `"logistic"` | Standard logistic regression (default) | Most cases |
| **Probit** | `"probit"` | Probit model (Φ⁻¹ transform) | Robustness check |
| **GBM** | `"gbm"` | Gradient boosting (HistGradientBoosting) | Non-linear relationships |
| **Random Forest** | `"rf"` | Random forest classifier | High-dimensional, interactions |
| **Lasso** | `"lasso"` | L1-regularized logistic | Variable selection |
| **Ridge** | `"ridge"` | L2-regularized logistic | Multicollinearity |
| **ElasticNet** | `"elasticnet"` | L1+L2 regularized logistic | Combined regularization |
| **Mahalanobis** | `"mahalanobis"` | No propensity model — covariate distance | Direct covariate matching |

```python
# Use gradient boosting for propensity scores
result = rollmatch(data, ..., model_type="gbm")

# Use Mahalanobis distance (no propensity model)
result = rollmatch(data, ..., model_type="mahalanobis")
```

## How It Works

### Rolling Entry Matching

For staggered adoption, each treated unit enters at a different time. Rolling entry matching:

1. **`reduce_data()`** — For each treated unit at entry time *t*, select covariates at *t − lookback*. Controls get one observation per treatment entry period.

2. **`score_data()`** — Fit propensity model, compute logit-transformed scores.

3. **`match_all_periods()`** — For each time period, match treated to closest controls within caliper. Uses **block-vectorized numpy broadcasting** (not full cross-join).

4. **`compute_balance()`** — Compare covariate means/SDs before and after matching.

### Key Optimization

R's bottleneck: `inner_join(treated, controls, by=time)` creates N_treated × N_controls rows.

pyrollmatch instead processes treated in **blocks** via numpy broadcasting:
```
Block of 2000 treated × 64K controls = 128M distances = ~1 GB
```
Never materializes the full cross-product as a DataFrame.

## Data Format

Input must be a `polars.DataFrame` in **long panel format** (one row per unit per time period):

| Column | Type | Description |
|---|---|---|
| `id` | int or str | Unit identifier (e.g., individual, firm, repository) |
| `tm` | int | Time period, must be integer and monotonically increasing (e.g., 1, 2, ..., 20) |
| `treat` | int (0 or 1) | **Time-invariant treatment group indicator.** `1` = unit that eventually receives treatment, `0` = unit that never receives treatment. This is NOT a time-varying treatment status — it labels the *group*, not whether treatment is currently active. |
| `entry` | int | **Treatment onset period** for treated units (the time period when treatment begins). For control units, set this to any value **strictly greater** than the maximum time period in your data. For example, if your panel spans periods 1–20, use `entry=99` or `entry=999` for controls. The specific value does not matter as long as it exceeds `max(tm)`. |
| covariates | float | Matching variables (e.g., pre-computed rolling means of activity measures) |

**Example:**
```
unit_id | time | treat | entry_time | x1   | x2
--------|------|-------|------------|------|-----
1       | 1    | 1     | 5          | 2.3  | 1.1   <- treated, enters at period 5
1       | 2    | 1     | 5          | 2.5  | 1.0
...
1       | 5    | 1     | 5          | 4.1  | 1.8   <- treatment starts here
...
101     | 1    | 0     | 99         | 1.8  | 0.9   <- control, entry=99 (sentinel)
101     | 2    | 0     | 99         | 1.9  | 1.0
```

## API Reference

### Core Functions

| Function | Description |
|---|---|
| `rollmatch()` | Full rolling entry matching pipeline |
| `alpha_sweep()` | Try multiple calipers, select best |
| `reduce_data()` | Construct quasi-panel for matching |
| `score_data()` | Compute propensity scores (8 models) |

### Diagnostics

| Function | Description |
|---|---|
| `balance_test()` | SMD + t-test + variance ratio + KS test |
| `equivalence_test()` | TOST equivalence test (Hartman & Hidalgo 2018) |
| `compute_balance()` | Covariate balance table (pooled across all periods) |
| `balance_by_period()` | Per-period SMD with aggregate summary |
| `smd_table()` | Print formatted SMD table |

### RollmatchResult

| Attribute | Type | Description |
|---|---|---|
| `matched_data` | `pl.DataFrame` | `treat_id`, `control_id`, `difference` |
| `balance` | `pl.DataFrame` | SMD for each covariate |
| `weights` | `pl.DataFrame` | `id`, `weight` |
| `n_treated_matched` | `int` | Number matched |
| `n_treated_total` | `int` | Total treated |
| `alpha` | `float` | Caliper used |

## Validation Against R rollmatch

Tested on identical synthetic data across 15 configurations:

| Metric | Result |
|---|---|
| Propensity score correlation | **≥ 0.9999** across all configs |
| Match count agreement | Within ±3% |
| Balance quality (max\|SMD\|) | Both achieve < 0.05 |

Pair-level overlap varies by caliper tightness (3%–71%) due to different GLM solvers (sklearn vs R glm). This is expected — **balance quality is what matters, not pair identity**.

> **Note on matching algorithm**: Our greedy matching processes treated units sequentially (per-treated nearest neighbor), while R rollmatch uses a global greedy approach (best-first across all treated). Both are valid greedy algorithms, but may produce different match assignments. Balance quality is comparable.

## Performance

| Scale | Runtime | Match rate |
|---|---:|---:|
| 500 × 2,000 | 0.03s | 100% |
| 1,000 × 3,000 | 0.05s | 100% |
| 5,000 × 15,000 | 0.10s | 100% |
| 10,000 × 30,000 | 1.7s | 99.99% |

R rollmatch crashes at 10K treated due to dplyr's 2.1B row limit.

## Design Principles

- **Polars-first**: Native polars DataFrames throughout. No pandas dependency.
- **Reproducible**: Deterministic with fixed `random_state`. Same input → same output.
- **Scalable**: Block-vectorized matching, O(block × N_controls) memory per iteration.
- **Validated**: 44 tests, R-validated across 15 synthetic configurations.
- **Modular**: Each step (reduce → score → match → balance) is independently usable.

## References

- Witman, A., Acquah, J., Alvelais, A., et al. (2018). "Comparison Group Selection in the Presence of Rolling Entry." *Health Services Research*, 54(1), 262–270. doi:10.1111/1475-6773.13086
- RTI International. `rollmatch` R package. https://github.com/RTIInternational/rollmatch
- Hartman, E., & Hidalgo, F. D. (2018). "An Equivalence Approach to Balance and Placebo Tests." *American Journal of Political Science*, 62(4), 1000–1013. doi:10.1111/ajps.12387

## Status & Contributing

This package is under **active development**. We welcome:

- **Bug reports** — if you encounter unexpected behavior, please open an issue with a minimal reproducible example
- **Feature requests** — suggestions for new matching methods, diagnostics, or API improvements
- **Contributions** — pull requests are welcome; please open an issue first to discuss

Please be cautious when using this package for published research. While we have validated against the R rollmatch package across multiple configurations (see [Validation](#validation-against-r-rollmatch)), this is alpha software. We strongly recommend:

1. **Cross-checking** results against an established implementation (R rollmatch, R MatchIt) for critical analyses
2. **Inspecting balance diagnostics** carefully before proceeding to estimation
3. **Reporting any discrepancies** you find via [GitHub Issues](https://github.com/AlanHuang99/pyrollmatch/issues)

## License

MIT
