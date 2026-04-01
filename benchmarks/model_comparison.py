"""
Model comparison: run all 8 scoring methods on synthetic data,
compare match quality, balance, and runtime.
"""

import polars as pl
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pyrollmatch import rollmatch
from pyrollmatch.score import SUPPORTED_MODELS

SEED = 42


def generate_data(n_treated=500, n_controls=2000, n_periods=15, n_covs=5):
    rng = np.random.default_rng(SEED)
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


def main():
    print("=" * 70)
    print("  Model Comparison: All 8 Scoring Methods")
    print("=" * 70)

    data = generate_data()
    covariates = [f"x{i+1}" for i in range(5)]

    print(f"\n  Data: {data.height} rows, {data['unit_id'].n_unique()} units")
    print(f"  Covariates: {covariates}")
    print(f"  Alpha: 0.1, Matches: 3\n")

    print(f"  {'Model':<15} {'Time':>8} {'Matched':>10} {'Pairs':>8} {'Max|SMD|':>10} {'Pass':>6}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*6}")

    results = []
    for model_type in SUPPORTED_MODELS:
        start = time.time()
        result = rollmatch(
            data, "treat", "time", "entry_time", "unit_id",
            covariates=covariates, alpha=0.1, num_matches=3,
            model_type=model_type, verbose=False,
        )
        elapsed = time.time() - start

        if result is not None:
            max_smd = result.balance["matched_smd"].abs().max()
            passed = max_smd < 0.1
            print(f"  {model_type:<15} {elapsed:>7.3f}s {result.n_treated_matched:>10} "
                  f"{result.matched_data.height:>8} {max_smd:>10.4f} {'✓' if passed else '✗':>6}")
            results.append({
                "model": model_type, "time": elapsed, "matched": result.n_treated_matched,
                "pairs": result.matched_data.height, "max_smd": max_smd, "pass": passed,
            })
        else:
            print(f"  {model_type:<15} {elapsed:>7.3f}s {'FAILED':>10}")

    # Summary
    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n  {n_pass}/{len(results)} models achieve |SMD| < 0.1")

    # Build HTML report
    rows_html = ""
    for r in results:
        cls = "pass" if r["pass"] else "fail"
        rows_html += f"""<tr>
            <td>{r['model']}</td><td>{r['time']:.3f}s</td>
            <td>{r['matched']}</td><td>{r['pairs']}</td>
            <td class="{cls}">{r['max_smd']:.4f}</td>
            <td class="{cls}">{'✓' if r['pass'] else '✗'}</td></tr>"""

    html = f"""<!DOCTYPE html>
<html><head><title>pyrollmatch Model Comparison</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: right; }}
th {{ background: #f3f4f6; }}
td:first-child {{ text-align: left; font-weight: 600; }}
.pass {{ color: #059669; font-weight: 600; }}
.fail {{ color: #dc2626; font-weight: 600; }}
</style></head><body>
<h1>pyrollmatch Model Comparison</h1>
<p>All 8 scoring methods on synthetic data (500 treated, 2000 controls, 5 covariates, α=0.1)</p>
<table>
<tr><th>Model</th><th>Runtime</th><th>Matched</th><th>Pairs</th><th>Max|SMD|</th><th>Pass</th></tr>
{rows_html}
</table>
<p><strong>{n_pass}/{len(results)} models pass</strong> the |SMD| &lt; 0.1 threshold.</p>
</body></html>"""

    report_path = Path(__file__).parent / "model_comparison_report.html"
    report_path.write_text(html)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
