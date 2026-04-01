"""
Extended validation: systematic comparison of pyrollmatch vs R rollmatch
across multiple configurations.

Tests:
  1. Different sample sizes (100, 500, 1000, 2000, 5000 treated)
  2. Different alpha values (0.001, 0.01, 0.05, 0.1, 0.2)
  3. Different num_matches (1, 3, 5)
  4. Different covariate counts (3, 5, 10)
  5. Replacement modes (cross_cohort vs R replacement=FALSE)

For each config, compares:
  - Score correlation
  - Match count difference
  - Max|SMD| difference
  - Pair overlap percentage

Outputs: benchmarks/extended_validation_report.html
"""

import polars as pl
import numpy as np
import subprocess
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pyrollmatch import rollmatch, reduce_data, score_data

REPORT_DIR = Path(__file__).parent
SEED = 42
R_LIB = str(Path.home() / "Dropbox/Workspace/AICoding/renv/library")
R_TIMEOUT = 120  # seconds


def generate_data(n_treated, n_controls, n_periods=15, n_covs=5, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated):
        entry_t = rng.integers(8, 13)
        base = rng.exponential(2.0, size=n_covs)
        for t in range(1, n_periods + 1):
            boost = 1.0 + 0.5 * (t >= entry_t)
            row = {"unit_id": i, "time": t, "treat": 1, "entry_time": int(entry_t)}
            for c in range(n_covs):
                row[f"x{c+1}"] = float(base[c] * boost + rng.normal(0, 0.3))
            rows.append(row)
    for i in range(n_controls):
        base = rng.exponential(2.0, size=n_covs)
        for t in range(1, n_periods + 1):
            row = {"unit_id": n_treated + i, "time": t, "treat": 0, "entry_time": 99}
            for c in range(n_covs):
                row[f"x{c+1}"] = float(base[c] + rng.normal(0, 0.3))
            rows.append(row)
    return pl.DataFrame(rows)


def run_py(data, covariates, alpha, num_matches, replacement=True):
    start = time.time()
    result = rollmatch(
        data, treat="treat", tm="time", entry="entry_time", id="unit_id",
        covariates=covariates, lookback=1, alpha=alpha,
        num_matches=num_matches, replacement=replacement, verbose=False,
    )
    elapsed = time.time() - start
    if result is None:
        return {"elapsed": elapsed, "n_matched": 0, "max_smd": None, "pairs": set(), "scores": {}}

    max_smd = result.balance["matched_smd"].abs().max()
    pairs = set(zip(
        result.matched_data["treat_id"].to_list(),
        result.matched_data["control_id"].to_list(),
    ))

    # Get scores
    reduced = reduce_data(data, "treat", "time", "entry_time", "unit_id", lookback=1)
    reduced = reduced.drop_nulls(subset=covariates)
    scored = score_data(reduced, covariates, "treat")
    score_map = dict(zip(scored["unit_id"].to_list(), scored["score"].to_list()))

    return {
        "elapsed": elapsed,
        "n_matched": result.n_treated_matched,
        "n_total": result.n_treated_total,
        "n_pairs": result.matched_data.height,
        "max_smd": max_smd,
        "pairs": pairs,
        "scores": score_map,
    }


def run_r(data, covariates, alpha, num_matches, replacement=True):
    tmp_data = REPORT_DIR / "_tmp_ext_data.parquet"
    tmp_result = REPORT_DIR / "_tmp_ext_result.json"
    data.write_parquet(tmp_data, compression="snappy")

    repl_str = "TRUE" if replacement else "FALSE"
    covs_str = ', '.join(f'"{c}"' for c in covariates)

    r_script = f"""
    lib_dirs <- list.dirs("{R_LIB}", recursive = TRUE)
    r_lib <- lib_dirs[grepl("x86_64", lib_dirs)][1]
    if (!is.na(r_lib)) .libPaths(c(r_lib, .libPaths()))

    suppressMessages(library(arrow))
    suppressMessages(library(rollmatch))
    suppressMessages(library(jsonlite))

    df <- read_parquet("{tmp_data}")
    covs <- c({covs_str})
    fm <- as.formula(paste("treat ~", paste(covs, collapse = " + ")))

    t0 <- proc.time()
    reduced <- reduce_data(data=as.data.frame(df), treat="treat", tm="time",
                           entry="entry_time", id="unit_id", lookback=1)
    scored <- score_data(reduced_data=reduced, model_type="logistic", match_on="logit",
                         fm=fm, treat="treat", tm="time", entry="entry_time", id="unit_id")
    output <- rollmatch(scored_data=scored, data=as.data.frame(df), treat="treat",
                        tm="time", entry="entry_time", id="unit_id", vars=covs,
                        lookback=1, alpha={alpha}, standard_deviation="average",
                        num_matches={num_matches}, replacement={repl_str})
    elapsed <- (proc.time() - t0)[["elapsed"]]

    bal <- output$balance
    bal_df <- data.frame(Variable=rownames(bal),
      Mean_T=bal[,"Matched Treatment Mean"], Mean_C=bal[,"Matched Comparison Mean"],
      SD_T=bal[,"Matched Treatment Std Dev"], SD_C=bal[,"Matched Comparison Std Dev"])
    bal_df$SD_pooled <- sqrt((bal_df$SD_T^2 + bal_df$SD_C^2) / 2)
    bal_df$SMD <- (bal_df$Mean_T - bal_df$Mean_C) / bal_df$SD_pooled

    result <- list(
      elapsed=elapsed,
      n_matched=length(unique(output$matched_data$treat_id)),
      n_pairs=nrow(output$matched_data),
      max_smd=max(abs(bal_df$SMD), na.rm=TRUE),
      treat_ids=output$matched_data$treat_id,
      control_ids=output$matched_data$control_id,
      score_ids=scored$unit_id,
      score_vals=scored$score
    )
    writeLines(toJSON(result, auto_unbox=FALSE), "{tmp_result}")
    """

    try:
        proc = subprocess.run(["Rscript", "-e", r_script],
                              capture_output=True, text=True, timeout=R_TIMEOUT)
    except subprocess.TimeoutExpired:
        tmp_data.unlink(missing_ok=True)
        return {"elapsed": R_TIMEOUT, "n_matched": 0, "timeout": True}

    tmp_data.unlink(missing_ok=True)

    if proc.returncode != 0:
        tmp_result.unlink(missing_ok=True)
        return {"elapsed": 0, "n_matched": 0, "error": proc.stderr[:100]}

    with open(tmp_result) as f:
        r = json.load(f)
    tmp_result.unlink(missing_ok=True)

    el = r["elapsed"][0] if isinstance(r["elapsed"], list) else r["elapsed"]
    pairs = set(zip(r["treat_ids"], r["control_ids"]))
    score_map = dict(zip(r["score_ids"], r["score_vals"]))

    return {
        "elapsed": el,
        "n_matched": r["n_matched"][0] if isinstance(r["n_matched"], list) else r["n_matched"],
        "n_pairs": r["n_pairs"][0] if isinstance(r["n_pairs"], list) else r["n_pairs"],
        "max_smd": r["max_smd"][0] if isinstance(r["max_smd"], list) else r["max_smd"],
        "pairs": pairs,
        "scores": score_map,
    }


def compare(py, r):
    """Compare Python and R results."""
    if not py["pairs"] or not r.get("pairs"):
        return {"score_corr": None, "pair_overlap": None}

    # Score correlation
    common_ids = sorted(set(py["scores"].keys()) & set(r["scores"].keys()))
    if common_ids:
        py_s = np.array([py["scores"][i] for i in common_ids])
        r_s = np.array([r["scores"][i] for i in common_ids])
        score_corr = np.corrcoef(py_s, r_s)[0, 1]
    else:
        score_corr = None

    # Pair overlap
    overlap = py["pairs"] & r["pairs"]
    denom = max(len(r["pairs"]), 1)
    pair_overlap_pct = 100 * len(overlap) / denom

    return {"score_corr": score_corr, "pair_overlap": pair_overlap_pct, "overlap_n": len(overlap)}


def main():
    print("=" * 70)
    print("  pyrollmatch Extended Validation: Python vs R")
    print("=" * 70)

    results = []

    # ── Test 1: Varying sample sizes ──
    print("\n── Test 1: Varying sample sizes (alpha=0.1, matches=3, covs=5) ──")
    for n_t in [100, 500, 1000, 2000]:
        n_c = n_t * 4
        label = f"n={n_t}"
        print(f"  {label}...", end=" ", flush=True)
        data = generate_data(n_t, n_c)
        covs = [f"x{i+1}" for i in range(5)]

        py = run_py(data, covs, 0.1, 3)
        r = run_r(data, covs, 0.1, 3)
        comp = compare(py, r)

        results.append({
            "test": "sample_size", "config": label,
            "py_time": py["elapsed"], "r_time": r.get("elapsed", None),
            "py_matched": py["n_matched"], "r_matched": r.get("n_matched", 0),
            "py_pairs": py["n_pairs"], "r_pairs": r.get("n_pairs", 0),
            "py_max_smd": py["max_smd"], "r_max_smd": r.get("max_smd", None),
            "score_corr": comp.get("score_corr"), "pair_overlap": comp.get("pair_overlap"),
        })
        print(f"Py={py['elapsed']:.2f}s R={r.get('elapsed', '?')}s "
              f"corr={comp.get('score_corr', '?'):.4f} "
              f"overlap={comp.get('pair_overlap', '?'):.1f}%"
              if comp.get('score_corr') else f"R failed: {r.get('error', 'timeout')}")

    # ── Test 2: Varying alpha ──
    print("\n── Test 2: Varying alpha (n=500, matches=3, covs=5) ──")
    data = generate_data(500, 2000)
    covs = [f"x{i+1}" for i in range(5)]
    for alpha in [0.001, 0.01, 0.05, 0.1, 0.2]:
        label = f"alpha={alpha}"
        print(f"  {label}...", end=" ", flush=True)

        py = run_py(data, covs, alpha, 3)
        r = run_r(data, covs, alpha, 3)
        comp = compare(py, r)

        results.append({
            "test": "alpha", "config": label,
            "py_time": py["elapsed"], "r_time": r.get("elapsed"),
            "py_matched": py["n_matched"], "r_matched": r.get("n_matched", 0),
            "py_pairs": py["n_pairs"], "r_pairs": r.get("n_pairs", 0),
            "py_max_smd": py["max_smd"], "r_max_smd": r.get("max_smd"),
            "score_corr": comp.get("score_corr"), "pair_overlap": comp.get("pair_overlap"),
        })
        print(f"Py={py['n_matched']} R={r.get('n_matched', '?')} "
              f"corr={comp.get('score_corr', 0):.4f} "
              f"overlap={comp.get('pair_overlap', 0):.1f}%"
              if comp.get("score_corr") else "R failed")

    # ── Test 3: Varying num_matches ──
    print("\n── Test 3: Varying num_matches (n=500, alpha=0.1, covs=5) ──")
    for nm in [1, 3, 5]:
        label = f"matches={nm}"
        print(f"  {label}...", end=" ", flush=True)

        py = run_py(data, covs, 0.1, nm)
        r = run_r(data, covs, 0.1, nm)
        comp = compare(py, r)

        results.append({
            "test": "num_matches", "config": label,
            "py_time": py["elapsed"], "r_time": r.get("elapsed"),
            "py_matched": py["n_matched"], "r_matched": r.get("n_matched", 0),
            "py_pairs": py["n_pairs"], "r_pairs": r.get("n_pairs", 0),
            "py_max_smd": py["max_smd"], "r_max_smd": r.get("max_smd"),
            "score_corr": comp.get("score_corr"), "pair_overlap": comp.get("pair_overlap"),
        })
        print(f"Py={py['n_pairs']} R={r.get('n_pairs', '?')} pairs, "
              f"overlap={comp.get('pair_overlap', 0):.1f}%")

    # ── Test 4: Varying covariates ──
    print("\n── Test 4: Varying covariate count (n=500, alpha=0.1, matches=3) ──")
    for nc in [3, 5, 10]:
        label = f"covs={nc}"
        print(f"  {label}...", end=" ", flush=True)
        data_nc = generate_data(500, 2000, n_covs=nc)
        covs_nc = [f"x{i+1}" for i in range(nc)]

        py = run_py(data_nc, covs_nc, 0.1, 3)
        r = run_r(data_nc, covs_nc, 0.1, 3)
        comp = compare(py, r)

        results.append({
            "test": "covariates", "config": label,
            "py_time": py["elapsed"], "r_time": r.get("elapsed"),
            "py_matched": py["n_matched"], "r_matched": r.get("n_matched", 0),
            "py_max_smd": py["max_smd"], "r_max_smd": r.get("max_smd"),
            "score_corr": comp.get("score_corr"), "pair_overlap": comp.get("pair_overlap"),
        })
        print(f"corr={comp.get('score_corr', 0):.4f} overlap={comp.get('pair_overlap', 0):.1f}%")

    # ── Test 5: Replacement=cross_cohort (maps to R replacement=FALSE) ──
    print("\n── Test 5: replacement='cross_cohort' (n=500, alpha=0.1, matches=1) ──")
    py = run_py(data, covs, 0.1, 1, replacement="cross_cohort")
    r = run_r(data, covs, 0.1, 1, replacement=False)
    comp = compare(py, r)
    results.append({
        "test": "cross_cohort", "config": "repl=cross_cohort",
        "py_time": py["elapsed"], "r_time": r.get("elapsed"),
        "py_matched": py["n_matched"], "r_matched": r.get("n_matched", 0),
        "py_max_smd": py["max_smd"], "r_max_smd": r.get("max_smd"),
        "score_corr": comp.get("score_corr"), "pair_overlap": comp.get("pair_overlap"),
    })
    print(f"corr={comp.get('score_corr', 0):.4f} overlap={comp.get('pair_overlap', 0):.1f}%")

    # ── Generate HTML Report ──
    print("\n── Generating report ──")
    html = build_html(results)
    report_path = REPORT_DIR / "extended_validation_report.html"
    report_path.write_text(html)
    print(f"  Saved: {report_path}")
    print("\nDone.")


def build_html(results):
    rows_html = ""
    for r in results:
        sc = f"{r['score_corr']:.6f}" if r.get('score_corr') is not None else "N/A"
        po = f"{r['pair_overlap']:.1f}%" if r.get('pair_overlap') is not None else "N/A"
        py_smd = f"{r['py_max_smd']:.4f}" if r.get('py_max_smd') is not None else "N/A"
        r_smd = f"{r['r_max_smd']:.4f}" if r.get('r_max_smd') is not None else "N/A"
        r_time = f"{r['r_time']:.2f}s" if r.get('r_time') else "N/A"

        smd_match = ""
        if r.get('py_max_smd') is not None and r.get('r_max_smd') is not None:
            both_pass = r['py_max_smd'] < 0.1 and r['r_max_smd'] < 0.1
            smd_match = "pass" if both_pass else "fail"

        rows_html += f"""
        <tr>
            <td>{r['test']}</td>
            <td>{r['config']}</td>
            <td>{r['py_time']:.3f}s</td>
            <td>{r_time}</td>
            <td>{r['py_matched']}</td>
            <td>{r.get('r_matched', 'N/A')}</td>
            <td class="{smd_match}">{py_smd}</td>
            <td class="{smd_match}">{r_smd}</td>
            <td>{sc}</td>
            <td>{po}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><title>pyrollmatch Extended Validation</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
th, td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: right; }}
th {{ background: #f3f4f6; font-weight: 600; }}
td:first-child, td:nth-child(2) {{ text-align: left; }}
.pass {{ color: #059669; }}
.fail {{ color: #dc2626; }}
</style></head><body>
<h1>pyrollmatch Extended Validation Report</h1>
<p>Systematic comparison of pyrollmatch (Python) vs rollmatch (R) across {len(results)} configurations.</p>
<table>
<tr>
  <th>Test</th><th>Config</th>
  <th>Py Time</th><th>R Time</th>
  <th>Py Matched</th><th>R Matched</th>
  <th>Py max|SMD|</th><th>R max|SMD|</th>
  <th>Score Corr</th><th>Pair Overlap</th>
</tr>
{rows_html}
</table>
<h2>Interpretation</h2>
<ul>
<li><strong>Score Correlation</strong>: Should be &gt; 0.99. Measures agreement of propensity score models.</li>
<li><strong>Pair Overlap</strong>: Varies by caliper tightness. Tight calipers &rarr; small differences in scores &rarr; different pairs.</li>
<li><strong>max|SMD|</strong>: Both should be &lt; 0.1 for valid matching. Agreement here matters more than pair identity.</li>
</ul>
<p style="color:#94a3b8; font-size:12px; margin-top:40px;">Generated by pyrollmatch extended validation. Seed={SEED}.</p>
</body></html>"""


if __name__ == "__main__":
    main()
