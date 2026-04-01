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


def balance_by_period(
    scored_data: pl.DataFrame,
    matches: pl.DataFrame,
    treat: str,
    id: str,
    tm: str,
    covariates: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute per-period covariate balance (SMD) for matched samples.

    Pooled SMD can mask within-cohort imbalance through cancellation.
    This function computes SMD separately for each entry period, then
    aggregates across periods.

    Parameters
    ----------
    scored_data : pl.DataFrame
        Reduced data with treatment indicator and covariates.
    matches : pl.DataFrame
        Match results with treat_id, control_id, and time period columns.
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
    (aggregate, detail) : tuple[pl.DataFrame, pl.DataFrame]

        **aggregate** — one row per covariate with columns:
        ``covariate``, ``wtd_mean_smd``, ``median_abs_smd``,
        ``max_abs_smd``, ``n_periods``.

        **detail** — one row per (period, covariate) with columns:
        ``period``, ``covariate``, ``n_treated``, ``n_controls``,
        ``mean_treated``, ``mean_control``, ``smd``.
    """
    time_periods = matches[tm].unique().sort().to_list()

    detail_rows = []

    for t in time_periods:
        period_matches = matches.filter(pl.col(tm) == t)

        # Treated units in this period
        t_ids = period_matches.select("treat_id").unique().rename({"treat_id": id})
        t_data = scored_data.filter(
            (pl.col(treat) == 1) & (pl.col(tm) == t)
        ).join(t_ids, on=id, how="semi")

        # Control units in this period
        c_ids = period_matches.select("control_id").unique().rename({"control_id": id})
        c_data = scored_data.filter(
            (pl.col(treat) == 0) & (pl.col(tm) == t)
        ).join(c_ids, on=id, how="semi")

        n_treated = t_data.height
        n_controls = c_data.height

        if n_treated < 2 or n_controls < 2:
            continue

        for cov in covariates:
            vals_t = t_data[cov].drop_nulls().to_numpy()
            vals_c = c_data[cov].drop_nulls().to_numpy()

            if len(vals_t) < 2 or len(vals_c) < 2:
                detail_rows.append({
                    "period": t, "covariate": cov,
                    "n_treated": n_treated, "n_controls": n_controls,
                    "mean_treated": np.nan, "mean_control": np.nan,
                    "smd": np.nan,
                })
                continue

            mean_t = np.mean(vals_t)
            mean_c = np.mean(vals_c)
            sd_t = np.std(vals_t, ddof=1)
            sd_c = np.std(vals_c, ddof=1)
            pooled = np.sqrt((sd_t**2 + sd_c**2) / 2)
            smd = (mean_t - mean_c) / pooled if pooled > 0 else np.nan

            detail_rows.append({
                "period": t, "covariate": cov,
                "n_treated": n_treated, "n_controls": n_controls,
                "mean_treated": round(mean_t, 4),
                "mean_control": round(mean_c, 4),
                "smd": round(smd, 4),
            })

    detail = pl.DataFrame(detail_rows) if detail_rows else pl.DataFrame(
        schema={"period": pl.Int64, "covariate": pl.Utf8,
                "n_treated": pl.Int64, "n_controls": pl.Int64,
                "mean_treated": pl.Float64, "mean_control": pl.Float64,
                "smd": pl.Float64}
    )

    if detail.height == 0:
        agg = pl.DataFrame(
            schema={"covariate": pl.Utf8, "wtd_mean_smd": pl.Float64,
                    "median_abs_smd": pl.Float64, "max_abs_smd": pl.Float64,
                    "n_periods": pl.UInt32}
        )
        return agg, detail

    # Aggregate: weighted mean (by n_treated), median |SMD|, max |SMD|
    agg_rows = []
    for cov in covariates:
        cov_detail = detail.filter(
            (pl.col("covariate") == cov) & pl.col("smd").is_not_nan()
        )
        if cov_detail.height == 0:
            agg_rows.append({
                "covariate": cov, "wtd_mean_smd": np.nan,
                "median_abs_smd": np.nan, "max_abs_smd": np.nan,
                "n_periods": 0,
            })
            continue

        smds = cov_detail["smd"].to_numpy()
        weights = cov_detail["n_treated"].to_numpy().astype(float)
        total_w = weights.sum()

        wtd_mean = float(np.average(smds, weights=weights)) if total_w > 0 else np.nan
        median_abs = float(np.median(np.abs(smds)))
        max_abs = float(np.max(np.abs(smds)))

        agg_rows.append({
            "covariate": cov,
            "wtd_mean_smd": round(wtd_mean, 4),
            "median_abs_smd": round(median_abs, 4),
            "max_abs_smd": round(max_abs, 4),
            "n_periods": cov_detail.height,
        })

    agg = pl.DataFrame(agg_rows)
    return agg, detail


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
