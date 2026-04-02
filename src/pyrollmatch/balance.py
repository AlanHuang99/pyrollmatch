"""
balance — Covariate balance computation and SMD table.

Provides both unweighted (for matching) and weighted (for entropy
balancing / IPW) balance diagnostics.
"""

import polars as pl
import numpy as np


# ---------------------------------------------------------------------------
# Weighted statistics helpers
# ---------------------------------------------------------------------------

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean: Σ(w·x) / Σw."""
    sw = weights.sum()
    if sw == 0:
        return np.nan
    return float(np.sum(weights * values) / sw)


def _weighted_std(values: np.ndarray, weights: np.ndarray, ddof: int = 1) -> float:
    """Bessel-corrected weighted standard deviation for reliability weights.

    Uses V1/V2 correction:  var = Σw·(x − μ_w)² / (V1 − ddof·V2/V1)
    where V1 = Σw, V2 = Σw².
    """
    v1 = weights.sum()
    v2 = np.sum(weights ** 2)
    if v1 == 0:
        return np.nan
    mu = np.sum(weights * values) / v1
    denom = v1 - ddof * v2 / v1
    if denom <= 0:
        return np.nan
    var = np.sum(weights * (values - mu) ** 2) / denom
    return float(np.sqrt(max(var, 0.0)))


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


def compute_balance_weighted(
    data: pl.DataFrame,
    weights: pl.DataFrame,
    treat: str,
    id: str,
    covariates: list[str],
) -> pl.DataFrame:
    """Compute covariate balance using weighted means and SDs.

    Parameters
    ----------
    data : pl.DataFrame
        Reduced data with treatment indicator and covariates.
    weights : pl.DataFrame
        Unit weights with columns [id, weight].
    treat : str
        Treatment indicator column.
    id : str
        Unit identifier column.
    covariates : list[str]
        Covariate column names.

    Returns
    -------
    pl.DataFrame with same schema as compute_balance():
        covariate, full_mean_t, full_mean_c, full_sd_t, full_sd_c,
        full_smd, matched_mean_t, matched_mean_c, matched_sd_t,
        matched_sd_c, matched_smd
    """
    # Full sample (unweighted)
    full_t = data.filter(pl.col(treat) == 1)
    full_c = data.filter(pl.col(treat) == 0)

    # Weighted sample: join with weights
    weighted = data.join(weights, on=id, how="inner")
    w_t = weighted.filter(pl.col(treat) == 1)
    w_c = weighted.filter(pl.col(treat) == 0)

    rows = []
    for cov in covariates:
        # Full sample (unweighted)
        vals_ft = full_t[cov].drop_nulls().to_numpy()
        vals_fc = full_c[cov].drop_nulls().to_numpy()
        full_mean_t = np.mean(vals_ft) if len(vals_ft) > 0 else np.nan
        full_mean_c = np.mean(vals_fc) if len(vals_fc) > 0 else np.nan
        full_sd_t = np.std(vals_ft, ddof=1) if len(vals_ft) > 1 else np.nan
        full_sd_c = np.std(vals_fc, ddof=1) if len(vals_fc) > 1 else np.nan
        full_pooled = np.sqrt((full_sd_t**2 + full_sd_c**2) / 2) if not (np.isnan(full_sd_t) or np.isnan(full_sd_c)) else np.nan
        full_smd = (full_mean_t - full_mean_c) / full_pooled if full_pooled and full_pooled > 0 else np.nan

        # Weighted sample
        wvals_t = w_t[cov].drop_nulls().to_numpy()
        wvals_c = w_c[cov].drop_nulls().to_numpy()
        ww_t = w_t.filter(pl.col(cov).is_not_null())["weight"].to_numpy()
        ww_c = w_c.filter(pl.col(cov).is_not_null())["weight"].to_numpy()

        m_mean_t = _weighted_mean(wvals_t, ww_t) if len(wvals_t) > 0 else np.nan
        m_mean_c = _weighted_mean(wvals_c, ww_c) if len(wvals_c) > 0 else np.nan
        m_sd_t = _weighted_std(wvals_t, ww_t) if len(wvals_t) > 1 else np.nan
        m_sd_c = _weighted_std(wvals_c, ww_c) if len(wvals_c) > 1 else np.nan
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


def balance_by_period_weighted(
    data: pl.DataFrame,
    weights: pl.DataFrame,
    treat: str,
    id: str,
    tm: str,
    covariates: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-period covariate balance using weighted means/SDs.

    Parameters
    ----------
    data : pl.DataFrame
        Reduced data.
    weights : pl.DataFrame
        Unit weights. Either stacked [tm, id, weight] (per-cohort
        weights) or collapsed [id, weight]. If stacked, per-cohort
        weights are used for each period's balance computation.
    treat, id, tm : str
        Column names.
    covariates : list[str]
        Covariate column names.

    Returns
    -------
    (aggregate, detail) with same schemas as balance_by_period().
    """
    # Detect stacked vs collapsed weights
    stacked = tm in weights.columns
    if stacked:
        weighted = data.join(weights, on=[tm, id], how="inner")
    else:
        weighted = data.join(weights, on=id, how="inner")
    time_periods = weighted[tm].unique().sort().to_list()

    detail_rows = []
    for t in time_periods:
        period_data = weighted.filter(pl.col(tm) == t)
        t_data = period_data.filter(pl.col(treat) == 1)
        c_data = period_data.filter(pl.col(treat) == 0)

        n_treated = t_data.height
        n_controls = c_data.height
        if n_treated < 2 or n_controls < 2:
            continue

        for cov in covariates:
            vals_t = t_data[cov].drop_nulls().to_numpy()
            vals_c = c_data[cov].drop_nulls().to_numpy()
            ww_t = t_data.filter(pl.col(cov).is_not_null())["weight"].to_numpy()
            ww_c = c_data.filter(pl.col(cov).is_not_null())["weight"].to_numpy()

            if len(vals_t) < 2 or len(vals_c) < 2:
                detail_rows.append({
                    "period": t, "covariate": cov,
                    "n_treated": n_treated, "n_controls": n_controls,
                    "mean_treated": np.nan, "mean_control": np.nan,
                    "smd": np.nan,
                })
                continue

            mean_t = _weighted_mean(vals_t, ww_t)
            mean_c = _weighted_mean(vals_c, ww_c)
            sd_t = _weighted_std(vals_t, ww_t)
            sd_c = _weighted_std(vals_c, ww_c)
            pooled = np.sqrt((sd_t**2 + sd_c**2) / 2) if not (np.isnan(sd_t) or np.isnan(sd_c)) else np.nan
            smd = (mean_t - mean_c) / pooled if pooled and pooled > 0 else np.nan

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
        period_weights = cov_detail["n_treated"].to_numpy().astype(float)
        total_w = period_weights.sum()

        wtd_mean = float(np.average(smds, weights=period_weights)) if total_w > 0 else np.nan
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
