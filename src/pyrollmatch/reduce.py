"""
reduce_data — Construct quasi-panel for rolling entry matching.

For each treated unit at entry time t, selects their covariates at t-lookback.
For controls, creates one observation per treatment entry period (so controls
can serve as matches for multiple cohorts).
"""

import polars as pl


def reduce_data(
    data: pl.DataFrame,
    treat: str,
    tm: str,
    entry: str,
    id: str,
    lookback: int = 1,
) -> pl.DataFrame:
    """Construct quasi-panel for rolling entry matching.

    Parameters
    ----------
    data : pl.DataFrame
        Panel data with treat, time, entry, id columns and covariates.
    treat : str
        Column name for binary treatment indicator (1=treated, 0=control).
    tm : str
        Column name for time period.
    entry : str
        Column name for entry period. For treated units, this is the time
        period when treatment begins (integer). For control units, use either:
        - ``None``/``null`` (recommended), or
        - Any integer larger than max(tm) (e.g., 99, 999)
        Control units' entry values are never used by the algorithm.
    id : str
        Column name for unit identifier.
    lookback : int
        Number of periods to look back from entry for baseline covariates.

    Returns
    -------
    pl.DataFrame
        Reduced dataset: treated at baseline + controls at all baseline periods.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")

    for col in [treat, tm, entry, id]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # Treatment set: treated units at their baseline period (entry - lookback)
    # Controls may have null/None entry — that's fine, they're filtered by treat==0
    treat_set = data.filter(
        (pl.col(treat) == 1)
        & pl.col(entry).is_not_null()
        & (pl.col(tm) == pl.col(entry) - lookback)
    )

    # Get unique baseline time periods from treated set
    baseline_periods = treat_set[tm].unique().to_list()

    # Control set: controls at those same time periods
    control_set = data.filter(
        (pl.col(treat) == 0) & (pl.col(tm).is_in(baseline_periods))
    )

    # Combine
    reduced = pl.concat([treat_set, control_set])

    return reduced
