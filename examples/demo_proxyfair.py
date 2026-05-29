"""Small reviewer-facing ProxyFair demo on synthetic data.

Runs a tiny synthetic benchmark once with vanilla GES and once with full ProxyFair,
then prints comparable diagnostics.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.synthetic_ges_runner import run_single_experiment


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_summary(title, row):
    print(f"\n{title}")
    print("-" * len(title))
    keys = [
        "algorithm",
        "lambda_f",
        "shd",
        "suppressed_proxy",
        "retained_legit",
        "proxy_load_pa_y_lower",
        "proxy_load_pa_y_upper",
        "pse_total_error",
        "final_cpdag",
    ]
    for key in keys:
        print(f"{key:24s}: {_fmt(row.get(key))}")


def main():
    common = {
        "archetype_id": 1,
        "N": 300,
        "phi_target": 0.5,
        "seed": 42,
        "n_bins": 3,
    }

    baseline = run_single_experiment(
        algorithm="vanilla_ges",
        lambda_f=0.0,
        **common,
    )
    proxyfair = run_single_experiment(
        algorithm="fair_mec_full",
        lambda_f=1.0,
        **common,
    )

    print("ProxyFair demo (synthetic, small-scale)")
    _print_summary("Vanilla GES", baseline)
    _print_summary("ProxyFair (hard+soft MEC)", proxyfair)


if __name__ == "__main__":
    main()
