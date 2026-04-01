"""
balance — Covariate balance computation and SMD table.
"""

import polars as pl
import numpy as np


def compute_balance(
    scored_data: pl.DataFrame,
    matches: pl.DataFrame,
    treat: str,
    id: str,
    tm: str,
    covariates: list[str],
) -> pl.DataFrame:
    """Compute covariate balance before and after matching.

    Returns a table with means, SDs, and SMDs for each covariate,
    both in the full sample and the matched sample.

    Parameters
    ----------
    scored_data : pl.DataFrame
        Reduced data with treatment indicator and covariates.
    matches : pl.DataFrame
        Match results with treat_id and control_id columns.
    treat : str
        Treatment indicator column.
    id : str
        Unit identifier column.
    tm : str
        Time period column.
    covariates : list[str]
        Covariate column names.

    Returns
    -------
    pl.DataFrame with columns:
        covariate, full_mean_t, full_mean_c, full_sd_t, full_sd_c,
        full_smd, matched_mean_t, matched_mean_c, matched_sd_t,
        matched_sd_c, matched_smd
    """
    # Pre-compute matched data ONCE (not per covariate)
    treat_matches = matches.select(tm, "treat_id").unique().rename({"treat_id": id})
    control_matches = matches.select(tm, "control_id").unique().rename({"control_id": id})
    matched_ids_df = pl.concat([treat_matches, control_matches])
    matched_data = scored_data.join(matched_ids_df, on=[tm, id], how="semi")

    # Pre-split by treatment group
    full_t = scored_data.filter(pl.col(treat) == 1)
    full_c = scored_data.filter(pl.col(treat) == 0)
    match_t = matched_data.filter(pl.col(treat) == 1)
    match_c = matched_data.filter(pl.col(treat) == 0)

    rows = []

    for cov in covariates:
        vals_t = full_t[cov].drop_nulls().to_numpy()
        vals_c = full_c[cov].drop_nulls().to_numpy()

        full_mean_t = np.mean(vals_t) if len(vals_t) > 0 else np.nan
        full_mean_c = np.mean(vals_c) if len(vals_c) > 0 else np.nan
        full_sd_t = np.std(vals_t, ddof=1) if len(vals_t) > 1 else np.nan
        full_sd_c = np.std(vals_c, ddof=1) if len(vals_c) > 1 else np.nan
        full_pooled = np.sqrt((full_sd_t**2 + full_sd_c**2) / 2) if not (np.isnan(full_sd_t) or np.isnan(full_sd_c)) else np.nan
        full_smd = (full_mean_t - full_mean_c) / full_pooled if full_pooled and full_pooled > 0 else np.nan

        mvals_t = match_t[cov].drop_nulls().to_numpy()
        mvals_c = match_c[cov].drop_nulls().to_numpy()

        m_mean_t = np.mean(mvals_t) if len(mvals_t) > 0 else np.nan
        m_mean_c = np.mean(mvals_c) if len(mvals_c) > 0 else np.nan
        m_sd_t = np.std(mvals_t, ddof=1) if len(mvals_t) > 1 else np.nan
        m_sd_c = np.std(mvals_c, ddof=1) if len(mvals_c) > 1 else np.nan
        m_pooled = np.sqrt((m_sd_t**2 + m_sd_c**2) / 2) if not (np.isnan(m_sd_t) or np.isnan(m_sd_c)) else np.nan
        m_smd = (m_mean_t - m_mean_c) / m_pooled if m_pooled and m_pooled > 0 else np.nan

        rows.append({
            "covariate": cov,
            "full_mean_t": round(full_mean_t, 4),
            "full_mean_c": round(full_mean_c, 4),
            "full_sd_t": round(full_sd_t, 4),
            "full_sd_c": round(full_sd_c, 4),
            "full_smd": round(full_smd, 4),
            "matched_mean_t": round(m_mean_t, 4),
            "matched_mean_c": round(m_mean_c, 4),
            "matched_sd_t": round(m_sd_t, 4),
            "matched_sd_c": round(m_sd_c, 4),
            "matched_smd": round(m_smd, 4),
        })

    return pl.DataFrame(rows)


def smd_table(balance: pl.DataFrame, threshold: float = 0.1) -> None:
    """Print a formatted SMD table with pass/fail indicators.

    Parameters
    ----------
    balance : pl.DataFrame
        Output from compute_balance().
    threshold : float
        |SMD| threshold for pass/fail (default 0.1).
    """
    max_smd = balance["matched_smd"].abs().max()
    all_pass = balance["matched_smd"].abs().max() < threshold

    print(f"\n{'='*70}")
    print(f"  Standardized Mean Differences (threshold: |SMD| < {threshold})")
    print(f"  Max |SMD| = {max_smd:.4f}  {'✓ ALL PASS' if all_pass else '✗ SOME FAIL'}")
    print(f"{'='*70}\n")
    print(f"  {'Covariate':<30} {'Full SMD':>10} {'Matched SMD':>12} {'Pass':>6}")
    print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*6}")

    for row in balance.iter_rows(named=True):
        smd = row["matched_smd"]
        passed = abs(smd) < threshold if smd is not None else False
        print(f"  {row['covariate']:<30} {row['full_smd']:>10.4f} {smd:>12.4f} {'✓' if passed else '✗':>6}")

    print()
