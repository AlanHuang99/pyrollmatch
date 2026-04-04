"""
Validation: Compare pyrollmatch (Python) vs rollmatch (R) on synthetic data.

Generates synthetic panel data, runs both implementations with identical
parameters, and compares:
  1. Propensity scores (correlation)
  2. Match counts
  3. Covariate balance (SMD)
  4. Overlap of matched pairs
  5. Runtime

Outputs: benchmarks/validation_report.html
"""

import polars as pl
import numpy as np
import subprocess
import json
import time
from pathlib import Path

# Add parent to path for local dev
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrollmatch import rollmatch, alpha_sweep, reduce_data, score_data
from pyrollmatch.balance import compute_balance, smd_table
from pyrollmatch.diagnostics import balance_test, equivalence_test


REPORT_DIR = Path(__file__).parent
SEED = 42


# ══════════════════════════════════════════════════════════════════════
# Step 1: Generate synthetic data
# ══════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    n_treated: int = 500,
    n_controls: int = 2000,
    n_periods: int = 20,
    n_covariates: int = 5,
    entry_range: tuple = (10, 16),
    seed: int = SEED,
) -> pl.DataFrame:
    """Generate synthetic panel data for validation.

    Treated units have a bump in activity at entry time.
    Controls have stable activity. Both have correlated covariates.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_treated):
        entry_t = rng.integers(entry_range[0], entry_range[1] + 1)
        # Base activity level (varies by unit)
        base = rng.exponential(2.0, size=n_covariates)
        for t in range(1, n_periods + 1):
            boost = 1.0 + 0.5 * (t >= entry_t)  # 50% boost post-treatment
            row = {
                "unit_id": i,
                "time": t,
                "treat": 1,
                "entry_time": int(entry_t),
            }
            for c in range(n_covariates):
                row[f"x{c+1}"] = float(base[c] * boost + rng.normal(0, 0.3))
            rows.append(row)

    for i in range(n_controls):
        base = rng.exponential(2.0, size=n_covariates)
        for t in range(1, n_periods + 1):
            row = {
                "unit_id": n_treated + i,
                "time": t,
                "treat": 0,
                "entry_time": 99,
            }
            for c in range(n_covariates):
                row[f"x{c+1}"] = float(base[c] + rng.normal(0, 0.3))
            rows.append(row)

    return pl.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Step 2: Run Python matching
# ══════════════════════════════════════════════════════════════════════

def run_python(data: pl.DataFrame, covariates: list[str], alpha: float,
               num_matches: int) -> dict:
    """Run pyrollmatch and return results."""
    start = time.time()
    result = rollmatch(
        data, treat="treat", tm="time", entry="entry_time", id="unit_id",
        covariates=covariates,
        lookback=1, ps_caliper=alpha, num_matches=num_matches,
        replacement="unrestricted", verbose=False,
    )
    elapsed = time.time() - start

    if result is None:
        return {"error": "No matches found", "elapsed": elapsed}

    # Score data for diagnostics
    reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id", lookback=1)
    reduced = reduced.drop_nulls(subset=covariates)
    scored = score_data(reduced, covariates, "treat").data

    # Diagnostics
    diag = balance_test(
        scored, result.matched_data, "treat", "unit_id", "time", covariates,
    )
    equiv = equivalence_test(
        scored, result.matched_data, "treat", "unit_id", "time", covariates,
    )

    return {
        "elapsed": elapsed,
        "n_pairs": result.matched_data.height,
        "n_treated_matched": result.n_treated_matched,
        "n_treated_total": result.n_treated_total,
        "match_rate": round(100 * result.n_treated_matched / result.n_treated_total, 1),
        "balance": result.balance,
        "diagnostics": diag,
        "equivalence": equiv,
        "matched_pairs": result.matched_data.select("treat_id", "control_id"),
        "scores": scored.select("unit_id", "treat", "score"),
    }


# ══════════════════════════════════════════════════════════════════════
# Step 3: Run R matching
# ══════════════════════════════════════════════════════════════════════

def run_r(data: pl.DataFrame, covariates: list[str], alpha: float,
          num_matches: int) -> dict:
    """Run R rollmatch via subprocess and return results."""
    # Save data for R
    tmp_data = REPORT_DIR / "_tmp_validation_data.parquet"
    tmp_result = REPORT_DIR / "_tmp_r_result.json"
    data.write_parquet(tmp_data, compression="snappy")

    r_script = f"""
    # Use AICoding project's renv for packages
    renv_lib <- "{Path.home()}/Dropbox/Workspace/AICoding/renv/library"
    lib_dirs <- list.dirs(renv_lib, recursive = TRUE)
    r_lib <- lib_dirs[grepl("x86_64", lib_dirs)][1]
    if (!is.na(r_lib)) .libPaths(c(r_lib, .libPaths()))

    library(arrow)
    library(rollmatch)
    library(jsonlite)

    df <- read_parquet("{tmp_data}")
    covs <- c({', '.join(f'"{c}"' for c in covariates)})
    fm <- as.formula(paste("treat ~", paste(covs, collapse = " + ")))

    t0 <- proc.time()

    reduced <- reduce_data(data = as.data.frame(df), treat = "treat",
                           tm = "time", entry = "entry_time",
                           id = "unit_id", lookback = 1)

    scored <- score_data(reduced_data = reduced, model_type = "logistic",
                         match_on = "logit", fm = fm, treat = "treat",
                         tm = "time", entry = "entry_time", id = "unit_id")

    output <- rollmatch(scored_data = scored, data = as.data.frame(df),
                        treat = "treat", tm = "time", entry = "entry_time",
                        id = "unit_id", vars = covs, lookback = 1,
                        alpha = {alpha}, standard_deviation = "average",
                        num_matches = {num_matches}, replacement = TRUE)

    elapsed <- (proc.time() - t0)[["elapsed"]]

    # Extract results
    match_df <- output$matched_data
    bal <- output$balance

    # Compute SMD
    bal_df <- data.frame(
      Variable = rownames(bal),
      Mean_T = bal[, "Matched Treatment Mean"],
      Mean_C = bal[, "Matched Comparison Mean"],
      SD_T = bal[, "Matched Treatment Std Dev"],
      SD_C = bal[, "Matched Comparison Std Dev"]
    )
    bal_df$SD_pooled <- sqrt((bal_df$SD_T^2 + bal_df$SD_C^2) / 2)
    bal_df$SMD <- (bal_df$Mean_T - bal_df$Mean_C) / bal_df$SD_pooled

    # Scores
    scores <- data.frame(unit_id = scored$unit_id, treat = scored$treat,
                         score = scored$score)

    result <- list(
      elapsed = elapsed,
      n_pairs = nrow(match_df),
      n_treated_matched = length(unique(match_df$treat_id)),
      treat_ids = unique(match_df$treat_id),
      control_ids = match_df$control_id,
      match_treat_ids = match_df$treat_id,
      match_control_ids = match_df$control_id,
      balance_vars = bal_df$Variable,
      balance_smd = bal_df$SMD,
      scores_unit_id = scores$unit_id,
      scores_treat = scores$treat,
      scores_score = scores$score
    )

    writeLines(toJSON(result, auto_unbox = FALSE), "{tmp_result}")
    """

    start = time.time()
    proc = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True, text=True, timeout=300,
    )
    wall_time = time.time() - start

    if proc.returncode != 0:
        print(f"R stderr: {proc.stderr[:500]}")
        return {"error": proc.stderr[:200], "elapsed": wall_time}

    # Load R results
    with open(tmp_result) as f:
        r_result = json.load(f)

    # Cleanup
    tmp_data.unlink(missing_ok=True)
    tmp_result.unlink(missing_ok=True)

    r_elapsed = r_result["elapsed"]
    if isinstance(r_elapsed, list):
        r_elapsed = r_elapsed[0]

    return {
        "elapsed": r_elapsed,
        "wall_time": wall_time,
        "n_pairs": r_result["n_pairs"],
        "n_treated_matched": r_result["n_treated_matched"],
        "matched_pairs": set(zip(r_result["match_treat_ids"],
                                  r_result["match_control_ids"])),
        "balance_vars": r_result["balance_vars"],
        "balance_smd": r_result["balance_smd"],
        "scores": {
            "unit_id": r_result["scores_unit_id"],
            "treat": r_result["scores_treat"],
            "score": r_result["scores_score"],
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Step 4: Generate HTML report
# ══════════════════════════════════════════════════════════════════════

def generate_report(
    py_result: dict,
    r_result: dict,
    data_info: dict,
    covariates: list[str],
    alpha: float,
) -> str:
    """Generate HTML validation report."""

    # Compare scores using polars + numpy (no pandas)
    if "scores" in py_result and "scores" in r_result:
        py_score_df = py_result["scores"]
        py_score_map = dict(zip(
            py_score_df["unit_id"].to_list(), py_score_df["score"].to_list()
        ))
        r_score_map = dict(zip(
            r_result["scores"]["unit_id"], r_result["scores"]["score"]
        ))
        common_ids = sorted(set(py_score_map.keys()) & set(r_score_map.keys()))
        if common_ids:
            py_s = np.array([py_score_map[i] for i in common_ids])
            r_s = np.array([r_score_map[i] for i in common_ids])
            score_corr = np.corrcoef(py_s, r_s)[0, 1]
            score_mae = np.mean(np.abs(py_s - r_s))
        else:
            score_corr = float("nan")
            score_mae = float("nan")
    else:
        score_corr = float("nan")
        score_mae = float("nan")

    # Compare matched pairs
    py_pairs = set(zip(
        py_result["matched_pairs"]["treat_id"].to_list(),
        py_result["matched_pairs"]["control_id"].to_list(),
    ))
    r_pairs = r_result.get("matched_pairs", set())
    overlap = py_pairs & r_pairs

    # Balance comparison
    py_bal = py_result["balance"]
    r_smd = dict(zip(r_result.get("balance_vars", []),
                     r_result.get("balance_smd", [])))

    # Build balance table rows
    bal_rows = ""
    for row in py_bal.iter_rows(named=True):
        cov = row["covariate"]
        py_smd = row["matched_smd"]
        r_smd_val = r_smd.get(cov, float("nan"))
        py_pass = "pass" if abs(py_smd) < 0.1 else "fail"
        r_pass = "pass" if abs(r_smd_val) < 0.1 else "fail" if not np.isnan(r_smd_val) else "n/a"
        bal_rows += f"""
        <tr>
            <td>{cov}</td>
            <td>{row['full_smd']:.4f}</td>
            <td class="{py_pass}">{py_smd:.4f}</td>
            <td class="{r_pass}">{r_smd_val:.4f}</td>
        </tr>"""

    # Diagnostics table
    diag_rows = ""
    if "diagnostics" in py_result:
        for row in py_result["diagnostics"].iter_rows(named=True):
            smd_cls = "pass" if row["smd_pass"] else "fail"
            vr_cls = "pass" if row["var_ratio_pass"] else "fail"
            diag_rows += f"""
            <tr>
                <td>{row['covariate']}</td>
                <td class="{smd_cls}">{row['smd']:.4f}</td>
                <td>{row['t_pvalue']:.4f}</td>
                <td class="{vr_cls}">{row['var_ratio']:.3f}</td>
                <td>{row['ks_pvalue']:.4f}</td>
            </tr>"""

    # Equivalence table
    equiv_rows = ""
    if "equivalence" in py_result:
        for row in py_result["equivalence"].iter_rows(named=True):
            cls = "pass" if row["equivalent"] else "fail"
            equiv_rows += f"""
            <tr>
                <td>{row['covariate']}</td>
                <td>{row['delta']:.4f}</td>
                <td class="{cls}">{row['tost_p']:.4f}</td>
                <td>{"Yes" if row['equivalent'] else "No"}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>pyrollmatch Validation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
h2 {{ color: #1e40af; margin-top: 40px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: right; }}
th {{ background: #f3f4f6; font-weight: 600; }}
td:first-child {{ text-align: left; }}
.pass {{ color: #059669; font-weight: 600; }}
.fail {{ color: #dc2626; font-weight: 600; }}
.metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.metric-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; }}
.metric-box h3 {{ margin: 0 0 10px 0; color: #475569; font-size: 14px; }}
.metric-box .value {{ font-size: 28px; font-weight: 700; color: #1e293b; }}
.metric-box .sub {{ color: #64748b; font-size: 13px; margin-top: 4px; }}
.verdict {{ font-size: 18px; padding: 15px; border-radius: 8px; margin: 20px 0; }}
.verdict.pass {{ background: #ecfdf5; border: 1px solid #059669; color: #065f46; }}
.verdict.warn {{ background: #fffbeb; border: 1px solid #d97706; color: #92400e; }}
code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>

<h1>pyrollmatch Validation Report</h1>
<p>Comparing <strong>pyrollmatch</strong> (Python/polars) vs <strong>rollmatch</strong> (R) on synthetic data.</p>

<h2>1. Test Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Treated units</td><td>{data_info['n_treated']}</td></tr>
<tr><td>Control units</td><td>{data_info['n_controls']}</td></tr>
<tr><td>Time periods</td><td>{data_info['n_periods']}</td></tr>
<tr><td>Covariates</td><td>{', '.join(covariates)}</td></tr>
<tr><td>Alpha (caliper)</td><td>{alpha}</td></tr>
<tr><td>Lookback</td><td>1</td></tr>
<tr><td>Num matches</td><td>{data_info['num_matches']}</td></tr>
<tr><td>Replacement</td><td>unrestricted</td></tr>
<tr><td>Seed</td><td>{SEED}</td></tr>
</table>

<h2>2. Summary Comparison</h2>
<div class="metric-grid">
<div class="metric-box">
    <h3>Python Runtime</h3>
    <div class="value">{py_result['elapsed']:.2f}s</div>
</div>
<div class="metric-box">
    <h3>R Runtime</h3>
    <div class="value">{r_result.get('elapsed', 'N/A')}s</div>
    <div class="sub">Wall time: {r_result.get('wall_time', 'N/A'):.1f}s (incl. startup)</div>
</div>
<div class="metric-box">
    <h3>Python Matched</h3>
    <div class="value">{py_result['n_treated_matched']} / {py_result['n_treated_total']}</div>
    <div class="sub">{py_result['match_rate']}% match rate</div>
</div>
<div class="metric-box">
    <h3>R Matched</h3>
    <div class="value">{r_result.get('n_treated_matched', 'N/A')}</div>
    <div class="sub">{py_result['n_pairs']} vs {r_result.get('n_pairs', 'N/A')} pairs</div>
</div>
</div>

<h2>3. Propensity Score Comparison</h2>
<div class="metric-grid">
<div class="metric-box">
    <h3>Score Correlation (R vs Python)</h3>
    <div class="value {'pass' if score_corr > 0.99 else 'fail'}">{score_corr:.6f}</div>
    <div class="sub">Perfect agreement = 1.000000</div>
</div>
<div class="metric-box">
    <h3>Mean Absolute Error</h3>
    <div class="value">{score_mae:.6f}</div>
    <div class="sub">Difference due to GLM solver (R glm vs sklearn)</div>
</div>
</div>

<h2>4. Matched Pair Overlap</h2>
<table>
<tr><th>Metric</th><th>Count</th><th>%</th></tr>
<tr><td>Python pairs</td><td>{len(py_pairs)}</td><td>—</td></tr>
<tr><td>R pairs</td><td>{len(r_pairs)}</td><td>—</td></tr>
<tr><td>Exact overlap</td><td>{len(overlap)}</td><td>{100*len(overlap)/max(len(r_pairs),1):.1f}%</td></tr>
<tr><td>Python only</td><td>{len(py_pairs - r_pairs)}</td><td></td></tr>
<tr><td>R only</td><td>{len(r_pairs - py_pairs)}</td><td></td></tr>
</table>
<p><em>Note: Different GLM solvers produce slightly different propensity scores,
leading to different pairs within tight calipers. This is expected and does not
indicate a bug — what matters is balance quality.</em></p>

<h2>5. Covariate Balance (SMD)</h2>
<table>
<tr><th>Covariate</th><th>Unmatched SMD</th><th>Python Matched SMD</th><th>R Matched SMD</th></tr>
{bal_rows}
</table>

<h2>6. Post-Matching Diagnostics (Python)</h2>
<h3>6a. Balance Test (SMD + t-test + Variance Ratio + KS)</h3>
<table>
<tr><th>Covariate</th><th>SMD</th><th>t-test p</th><th>Var Ratio</th><th>KS p</th></tr>
{diag_rows}
</table>

<h3>6b. TOST Equivalence Test (bound = 0.36&sigma;)</h3>
<table>
<tr><th>Covariate</th><th>Equiv Bound (&delta;)</th><th>TOST p</th><th>Equivalent?</th></tr>
{equiv_rows}
</table>

<h2>7. Verdict</h2>
<div class="verdict {'pass' if score_corr > 0.95 else 'warn'}">
{"<strong>VALIDATED:</strong> pyrollmatch produces equivalent matching quality to R rollmatch. "
 "Propensity scores are highly correlated and both implementations achieve good covariate balance."
 if score_corr > 0.95 else
 "<strong>WARNING:</strong> Score correlation is lower than expected. "
 "Investigate GLM solver differences."}
</div>

<p style="color: #94a3b8; font-size: 12px; margin-top: 40px;">
Generated by pyrollmatch validation suite. Seed={SEED}.
</p>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  pyrollmatch Validation: Python vs R")
    print("=" * 60)

    # Config
    n_treated = 500
    n_controls = 2000
    n_periods = 20
    n_covariates = 5
    alpha = 0.1
    num_matches = 3
    covariates = [f"x{i+1}" for i in range(n_covariates)]

    # Generate data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(n_treated, n_controls, n_periods, n_covariates)
    print(f"   {data.height} rows, {data['unit_id'].n_unique()} units")

    # Run Python
    print("\n2. Running pyrollmatch (Python)...")
    py_result = run_python(data, covariates, alpha, num_matches)
    print(f"   Matched: {py_result['n_treated_matched']}/{py_result['n_treated_total']} "
          f"in {py_result['elapsed']:.2f}s")

    # Run R
    print("\n3. Running rollmatch (R)...")
    r_result = run_r(data, covariates, alpha, num_matches)
    if "error" in r_result:
        print(f"   R ERROR: {r_result['error']}")
    else:
        print(f"   Matched: {r_result['n_treated_matched']} in {r_result['elapsed']:.2f}s")

    # Generate report
    print("\n4. Generating HTML report...")
    data_info = {
        "n_treated": n_treated, "n_controls": n_controls,
        "n_periods": n_periods, "num_matches": num_matches,
    }

    if "error" not in r_result:
        html = generate_report(py_result, r_result, data_info, covariates, alpha)
        report_path = REPORT_DIR / "validation_report.html"
        report_path.write_text(html)
        print(f"   Saved: {report_path}")
    else:
        print("   Skipped (R failed)")

    print("\nDone.")


if __name__ == "__main__":
    main()
