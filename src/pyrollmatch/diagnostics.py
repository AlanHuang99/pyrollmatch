"""
diagnostics — Post-matching diagnostic tests.

Includes t-tests, SMD tests, variance ratio tests, and equivalence tests
for assessing matching quality.
"""

import numpy as np
import polars as pl
from scipy import stats


def balance_test(
    scored_data: pl.DataFrame,
    matches: pl.DataFrame,
    treat: str,
    id: str,
    tm: str,
    covariates: list[str],
    threshold: float = 0.1,
) -> pl.DataFrame:
    """Run comprehensive balance diagnostics on matched sample.

    For each covariate, computes:
    - Standardized mean difference (SMD)
    - Two-sample t-test (H0: means are equal)
    - Variance ratio (treat/control)
    - Kolmogorov-Smirnov test (H0: distributions are equal)

    Parameters
    ----------
    scored_data : pl.DataFrame
        Reduced data with treatment indicator and covariates.
    matches : pl.DataFrame
        Match results with treat_id, control_id, tm columns.
    treat : str
        Treatment indicator column.
    id : str
        Unit identifier column.
    tm : str
        Time period column.
    covariates : list[str]
        Covariate column names.
    threshold : float
        SMD threshold for pass/fail (default 0.1).

    Returns
    -------
    pl.DataFrame with diagnostics per covariate.
    """
    # Get matched units
    treat_matches = matches.select(tm, "treat_id").unique().rename({"treat_id": id})
    control_matches = matches.select(tm, "control_id").unique().rename({"control_id": id})
    matched_ids = pl.concat([treat_matches, control_matches])
    matched_data = scored_data.join(matched_ids, on=[tm, id], how="semi")

    rows = []
    for cov in covariates:
        vals_t = matched_data.filter(pl.col(treat) == 1)[cov].drop_nulls().to_numpy().astype(float)
        vals_c = matched_data.filter(pl.col(treat) == 0)[cov].drop_nulls().to_numpy().astype(float)

        if len(vals_t) < 2 or len(vals_c) < 2:
            continue

        # SMD
        sd_t, sd_c = np.std(vals_t, ddof=1), np.std(vals_c, ddof=1)
        pooled_sd = np.sqrt((sd_t**2 + sd_c**2) / 2)
        smd = (np.mean(vals_t) - np.mean(vals_c)) / pooled_sd if pooled_sd > 0 else np.nan

        # Two-sample t-test (Welch's)
        t_stat, t_pvalue = stats.ttest_ind(vals_t, vals_c, equal_var=False)

        # Variance ratio
        var_ratio = np.var(vals_t, ddof=1) / np.var(vals_c, ddof=1) if np.var(vals_c, ddof=1) > 0 else np.nan

        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(vals_t, vals_c)

        rows.append({
            "covariate": cov,
            "mean_treated": round(np.mean(vals_t), 4),
            "mean_control": round(np.mean(vals_c), 4),
            "smd": round(smd, 4),
            "smd_pass": bool(abs(smd) < threshold),
            "t_stat": round(t_stat, 4),
            "t_pvalue": round(t_pvalue, 4),
            "var_ratio": round(var_ratio, 4),
            "var_ratio_pass": bool(0.5 < var_ratio < 2.0) if not np.isnan(var_ratio) else False,
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
        })

    result = pl.DataFrame(rows)

    # Print summary
    n_pass_smd = result.filter(pl.col("smd_pass")).height
    n_pass_var = result.filter(pl.col("var_ratio_pass")).height
    n_total = result.height

    print(f"\n{'='*70}")
    print(f"  Post-Matching Balance Diagnostics")
    print(f"{'='*70}")
    print(f"  SMD < {threshold}: {n_pass_smd}/{n_total} pass")
    print(f"  Variance ratio in (0.5, 2.0): {n_pass_var}/{n_total} pass")
    print(f"{'='*70}\n")

    print(f"  {'Covariate':<25} {'SMD':>8} {'t-test p':>10} {'VR':>8} {'KS p':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for row in result.iter_rows(named=True):
        smd_flag = "✓" if row["smd_pass"] else "✗"
        vr_flag = "✓" if row["var_ratio_pass"] else "✗"
        print(f"  {row['covariate']:<25} {row['smd']:>7.4f}{smd_flag} {row['t_pvalue']:>10.4f} {row['var_ratio']:>7.3f}{vr_flag} {row['ks_pvalue']:>8.4f}")

    return result


def equivalence_test(
    scored_data: pl.DataFrame,
    matches: pl.DataFrame,
    treat: str,
    id: str,
    tm: str,
    covariates: list[str],
    multiplier: float = 0.36,
) -> pl.DataFrame:
    """TOST equivalence test for covariate balance.

    Tests H0: |SMD| >= delta (non-equivalence).
    Rejection = GOOD (positive evidence of negligible difference).
    Uses Hartman & Hidalgo (2018) approach: delta = multiplier × pooled_SD.

    Parameters
    ----------
    scored_data : pl.DataFrame
        Reduced data.
    matches : pl.DataFrame
        Match results.
    treat, id, tm : str
        Column names.
    covariates : list[str]
        Covariate names.
    multiplier : float
        Equivalence bound as fraction of pooled SD (default 0.36).

    Returns
    -------
    pl.DataFrame with TOST results per covariate.
    """
    treat_matches = matches.select(tm, "treat_id").unique().rename({"treat_id": id})
    control_matches = matches.select(tm, "control_id").unique().rename({"control_id": id})
    matched_ids = pl.concat([treat_matches, control_matches])
    matched_data = scored_data.join(matched_ids, on=[tm, id], how="semi")

    rows = []
    for cov in covariates:
        vals_t = matched_data.filter(pl.col(treat) == 1)[cov].drop_nulls().to_numpy().astype(float)
        vals_c = matched_data.filter(pl.col(treat) == 0)[cov].drop_nulls().to_numpy().astype(float)

        if len(vals_t) < 2 or len(vals_c) < 2:
            continue

        m, n = len(vals_t), len(vals_c)
        diff = np.mean(vals_t) - np.mean(vals_c)
        var_t = np.var(vals_t, ddof=1)
        var_c = np.var(vals_c, ddof=1)

        # Pooled SD: weighted formula matching Hartman & Hidalgo (2018)
        # equivtest R package: sqrt(((m-1)*var_x + (n-1)*var_y) / (m+n-2))
        pooled_sd = np.sqrt(((m - 1) * var_t + (n - 1) * var_c) / (m + n - 2))
        delta = multiplier * pooled_sd

        # Two one-sided t-tests following equivtest::tost()
        # Uses Welch's t-test (unequal variances)
        se = np.sqrt(var_t / m + var_c / n)
        df_welch = se**4 / ((var_t/m)**2/(m-1) + (var_c/n)**2/(n-1)) if se > 0 else 1

        # Upper test: H0: diff >= delta, alt: diff < delta
        t_upper = (diff - delta) / se if se > 0 else np.inf
        p_upper = stats.t.cdf(t_upper, df=df_welch)

        # Lower test: H0: diff <= -delta, alt: diff > -delta
        t_lower = (diff + delta) / se if se > 0 else -np.inf
        p_lower = 1 - stats.t.cdf(t_lower, df=df_welch)

        tost_p = max(p_upper, p_lower)

        rows.append({
            "covariate": cov,
            "diff": round(diff, 6),
            "se": round(se, 6),
            "delta": round(delta, 4),
            "tost_p_upper": round(p_upper, 4),
            "tost_p_lower": round(p_lower, 4),
            "tost_p": round(tost_p, 4),
            "equivalent": bool(tost_p < 0.05),
        })

    result = pl.DataFrame(rows)

    n_equiv = result.filter(pl.col("equivalent")).height
    print(f"\n  TOST Equivalence Test (bound = {multiplier}σ)")
    print(f"  Equivalent: {n_equiv}/{result.height} covariates (p < 0.05 = GOOD)")
    for row in result.iter_rows(named=True):
        flag = "✓ EQUIV" if row["equivalent"] else "  not equiv"
        print(f"    {row['covariate']:<25} p={row['tost_p']:.4f} {flag}")

    return result
