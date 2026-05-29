# poetry run python src/experiments/run_synthetic_main_grid.py --mode result1 --n-jobs -1
# poetry run python run_synthetic_main_grid.py --mode result1 --plot-tradeoff --n-jobs -1 --disable-temporal-blacklist

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Ensure project root is importable when running this file directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.experiments.synthetic_ges_runner import run_single_experiment, save_results
from src.experiments.plot_synthetic_tradeoff import (
    plot_modal_graph_frequency_heatmap,
    plot_phi_sweep_paper_row,
    plot_phi_sweep_fairness_lines,
    plot_selectivity_curve,
)


try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None


ALGORITHMS = ["vanilla_ges", "hard_constraints", "fair_mec_phase1", "fair_mec_full", "fair_mec_marginal_only"]
LAMBDA_SWEEP = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
RESULT1_PHI_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RESULT1_FULL_LAMBDAS = [0.1, 0.5, 1.0, 2.0]
RESULT1_SOFT_ONLY_LAMBDA = 0.1


def _suppress_known_third_party_warnings() -> None:
    """Silence noisy pandas FutureWarnings emitted by upstream causallearn internals."""
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"causallearn\.score\.LocalScoreFunction",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"DataFrameGroupBy\.apply operated on the grouping columns.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group.*",
        category=FutureWarning,
    )


def _run_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    # joblib workers are separate processes; configure warning filters in each worker.
    _suppress_known_third_party_warnings()
    return run_single_experiment(**task)


def _grid_task(
    *,
    archetype_id: int,
    N: int,
    phi_target: float,
    seed: int,
    algorithm: str,
    lambda_f: float,
    n_bins: int = 5,
    confounding_strength: float | None = None,
    penalty_protected_indices: List[int] | None = None,
    condition: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    task: Dict[str, Any] = {
        "archetype_id": int(archetype_id),
        "N": int(N),
        "phi_target": float(phi_target),
        "seed": int(seed),
        "algorithm": str(algorithm),
        "lambda_f": float(lambda_f),
        "n_bins": int(n_bins),
    }
    if confounding_strength is not None:
        task["confounding_strength"] = float(confounding_strength)
    if penalty_protected_indices is not None:
        task["penalty_protected_indices"] = [int(i) for i in penalty_protected_indices]
    if condition is not None:
        task["condition"] = str(condition)
    if metadata:
        task["metadata"] = dict(metadata)
    return task


def _task_runner_kwargs(task: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(task)
    kwargs.pop("metadata", None)
    return kwargs


def _main_grid_tasks(smoke: bool) -> List[Dict[str, Any]]:
    archetypes = [1] if smoke else [1, 3]
    n_values = [500] if smoke else [500, 2000, 5000]
    phi_targets = [0.5] if smoke else [0.1, 0.5, 0.9]
    seeds = list(range(3)) if smoke else list(range(50))
    lambda_values = [0.1] if smoke else LAMBDA_SWEEP

    tasks: List[Dict[str, Any]] = []
    for archetype_id in archetypes:
        for N in n_values:
            for phi_target in phi_targets:
                for seed in seeds:
                    for algorithm in ALGORITHMS:
                        for lambda_f in lambda_values:
                            if algorithm in {"vanilla_ges", "hard_constraints"} and lambda_f != lambda_values[0]:
                                continue
                            tasks.append(
                                _grid_task(
                                    archetype_id=archetype_id,
                                    N=N,
                                    phi_target=phi_target,
                                    seed=seed,
                                    algorithm=algorithm,
                                    lambda_f=lambda_f,
                                )
                            )
    return tasks


def _figure1_tasks(smoke: bool) -> List[Dict[str, Any]]:
    archetypes = [1] if smoke else [1, 3]
    seeds = list(range(2)) if smoke else list(range(10))
    tasks: List[Dict[str, Any]] = []

    for archetype_id in archetypes:
        for seed in seeds:
            for baseline_alg in ["vanilla_ges", "hard_constraints"]:
                tasks.append(
                    _grid_task(
                        archetype_id=archetype_id,
                        N=5000,
                        phi_target=0.5,
                        seed=seed,
                        algorithm=baseline_alg,
                        lambda_f=0.0,
                    )
                )
            for fair_alg in ["fair_mec_full", "fair_mec_marginal_only"]:
                for lambda_f in LAMBDA_SWEEP:
                    tasks.append(
                        _grid_task(
                            archetype_id=archetype_id,
                            N=5000,
                            phi_target=0.5,
                            seed=seed,
                            algorithm=fair_alg,
                            lambda_f=lambda_f,
                        )
                    )

    return tasks


def _scenario_a_tasks(smoke: bool) -> List[Dict[str, Any]]:
    phi_targets = [0.05, 0.5, 0.95] if smoke else list(np.arange(0.05, 1.0, 0.05))
    seeds = list(range(3)) if smoke else list(range(50))

    tasks: List[Dict[str, Any]] = []
    for phi_target in phi_targets:
        for seed in seeds:
            for algorithm in ALGORITHMS:
                tasks.append(
                    _grid_task(
                        archetype_id=1,
                        N=5000,
                        phi_target=float(phi_target),
                        seed=seed,
                        algorithm=algorithm,
                        lambda_f=0.1,
                    )
                )
    return tasks


def _scenario_b_tasks(smoke: bool) -> List[Dict[str, Any]]:
    strengths = [0.3, 0.7] if smoke else [0.1, 0.3, 0.5, 0.7, 0.9]
    seeds = list(range(3)) if smoke else list(range(50))
    algorithms = ["fair_mec_full", "fair_mec_marginal_only", "vanilla_ges", "hard_constraints"]

    tasks: List[Dict[str, Any]] = []
    for strength in strengths:
        for seed in seeds:
            for algorithm in algorithms:
                tasks.append(
                    _grid_task(
                        archetype_id=3,
                        N=5000,
                        phi_target=0.5,
                        seed=seed,
                        algorithm=algorithm,
                        lambda_f=0.1,
                        confounding_strength=float(strength),
                        metadata={"confounding_strength": float(strength)},
                    )
                )
    return tasks


def _ablation_tasks(smoke: bool) -> List[Dict[str, Any]]:
    phi_targets = [0.1, 0.9] if smoke else [0.1, 0.5, 0.9]
    seeds = list(range(3)) if smoke else list(range(50))
    variant_map = {
        "phase1_only": "fair_mec_phase1",
        "phase2_only": "hard_constraints",
        "full": "fair_mec_full",
    }

    tasks: List[Dict[str, Any]] = []
    for variant, algorithm in variant_map.items():
        for phi_target in phi_targets:
            for seed in seeds:
                tasks.append(
                    _grid_task(
                        archetype_id=1,
                        N=5000,
                        phi_target=float(phi_target),
                        seed=seed,
                        algorithm=algorithm,
                        lambda_f=0.1,
                        metadata={"variant": variant},
                    )
                )

    return tasks


def _result1_tasks(smoke: bool) -> List[Dict[str, Any]]:
    # Result 1: Archetype A proxy edge suppression trade-off.
    # Fixed design: archetype=1, N=5000, phi sweep requested for proxy-strength analysis.
    phi_targets = RESULT1_PHI_SWEEP if not smoke else [0.1, 0.5, 0.9]
    seeds = list(range(3)) if smoke else list(range(50))

    tasks: List[Dict[str, Any]] = []
    for phi_target in phi_targets:
        for seed in seeds:
            for baseline_alg in ["vanilla_ges", "hard_constraints"]:
                tasks.append(
                    _grid_task(
                        archetype_id=1,
                        N=5000,
                        phi_target=phi_target,
                        seed=seed,
                        algorithm=baseline_alg,
                        lambda_f=0.0,
                    )
                )
            # Soft-only fairness stage.
            tasks.append(
                _grid_task(
                    archetype_id=1,
                    N=5000,
                    phi_target=phi_target,
                    seed=seed,
                    algorithm="fair_mec_phase1",
                    lambda_f=float(RESULT1_SOFT_ONLY_LAMBDA),
                )
            )
            # Full pipeline (hard + soft) across fairness penalties.
            for lambda_f in RESULT1_FULL_LAMBDAS:
                tasks.append(
                    _grid_task(
                        archetype_id=1,
                        N=5000,
                        phi_target=phi_target,
                        seed=seed,
                        algorithm="fair_mec_full",
                        lambda_f=float(lambda_f),
                    )
                )

    return tasks


def _displacement_tasks(smoke: bool) -> List[Dict[str, Any]]:
    seeds = list(range(3)) if smoke else list(range(50))
    specs = [
        ("baseline", "vanilla_ges", 0.0, None),
        ("penalize_s1", "fair_mec_full", 1.0, [1]),
        ("penalize_s2", "fair_mec_full", 1.0, [2]),
        ("penalize_joint", "fair_mec_full", 1.0, [1, 2]),
    ]

    tasks: List[Dict[str, Any]] = []
    for seed in seeds:
        for condition, algorithm, lambda_f, penalty_indices in specs:
            tasks.append(
                _grid_task(
                    archetype_id=4,
                    N=5000,
                    phi_target=0.5,
                    seed=seed,
                    algorithm=algorithm,
                    lambda_f=lambda_f,
                    n_bins=5,
                    penalty_protected_indices=penalty_indices,
                    condition=condition,
                    metadata={"condition": condition},
                )
            )
    return tasks


def _build_tasks_for_mode(mode: str, smoke: bool) -> List[Dict[str, Any]]:
    mode_builders = {
        "main_grid": _main_grid_tasks,
        "figure1": _figure1_tasks,
        "scenario_a": _scenario_a_tasks,
        "scenario_b": _scenario_b_tasks,
        "ablation_ges": _ablation_tasks,
        "result1": _result1_tasks,
        "displacement": _displacement_tasks,
    }
    if mode not in mode_builders:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode_builders[mode](smoke)



def _run_rows(tasks: List[Dict[str, Any]], n_jobs: int) -> List[Dict[str, Any]]:
    run_tasks = [_task_runner_kwargs(task) for task in tasks]
    _suppress_known_third_party_warnings()
    if Parallel is None or delayed is None or n_jobs == 1:
        return [_run_single_task(task) for task in run_tasks]
    return Parallel(n_jobs=n_jobs)(delayed(_run_single_task)(task) for task in run_tasks)



def build_tasks(
    smoke: bool,
    mode: str = "main_grid",
    enable_temporal_blacklist: bool = True,
    n_bins: int = 5,
) -> List[Dict[str, Any]]:
    tasks = _build_tasks_for_mode(mode=mode, smoke=smoke)
    for task in tasks:
        task["enable_temporal_blacklist"] = bool(enable_temporal_blacklist)
        task["n_bins"] = int(n_bins)
    return tasks


def _attach_task_metadata(rows: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> None:
    for row, task in zip(rows, tasks):
        metadata = task.get("metadata")
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            row[key] = value


def _print_phi_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    required = {"phi_target", "calibration_phi_r2_achieved", "calibration_phi_nmi_achieved"}
    if not required.issubset(df.columns):
        return
    grouped = (
        df.groupby(["phi_target"], dropna=False)[
            ["calibration_phi_r2_achieved", "calibration_phi_nmi_achieved"]
        ]
        .mean()
        .reset_index()
        .sort_values(["phi_target"])
    )
    print("Smoke calibration summary (means by phi_target):")
    for _, row in grouped.iterrows():
        print(
            "- phi={phi:.3f}, r2={r2:.4f}, nmi={nmi:.4f}".format(
                phi=float(row["phi_target"]),
                r2=float(row["calibration_phi_r2_achieved"]),
                nmi=float(row["calibration_phi_nmi_achieved"]),
            )
        )


def _print_phi_per_seed(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    required = {
        "seed",
        "phi_target",
        "calibration_phi_r2_achieved",
        "calibration_phi_nmi_achieved",
        "algorithm",
    }
    if not required.issubset(df.columns):
        return
    cols = [
        "seed",
        "algorithm",
        "phi_target",
        "calibration_phi_r2_achieved",
        "calibration_phi_nmi_achieved",
    ]
    print("Per-seed calibration diagnostics:")
    for _, row in df[cols].sort_values(["seed", "phi_target", "algorithm"]).iterrows():
        print(
            "- seed={seed}, alg={alg}, phi={phi:.3f}, r2={r2:.4f}, nmi={nmi:.4f}".format(
                seed=int(row["seed"]),
                alg=str(row["algorithm"]),
                phi=float(row["phi_target"]),
                r2=float(row["calibration_phi_r2_achieved"]),
                nmi=float(row["calibration_phi_nmi_achieved"]),
            )
        )


def _print_phi_star_cramers_validation(rows: List[Dict[str, Any]], tolerance: float = 0.02) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    required = {"phi_star", "seed", "calibration_phi_star_cramers_v_achieved"}
    if not required.issubset(df.columns):
        return

    check_df = df[list(required)].dropna().drop_duplicates(subset=["phi_star", "seed"])
    if check_df.empty:
        return
    check_df["abs_error"] = (check_df["calibration_phi_star_cramers_v_achieved"] - check_df["phi_star"]).abs()

    grouped = (
        check_df.groupby("phi_star", as_index=False)
        .agg(
            mean_achieved=("calibration_phi_star_cramers_v_achieved", "mean"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
            n_seed=("seed", "nunique"),
        )
        .sort_values("phi_star")
    )
    print(f"Cramer's V calibration check (tolerance +/-{tolerance:.3f}):")
    for _, row in grouped.iterrows():
        phi = float(row["phi_star"])
        achieved = float(row["mean_achieved"])
        mean_abs_error = float(row["mean_abs_error"])
        max_abs_error = float(row["max_abs_error"])
        n_seed = int(row["n_seed"])
        print(
            "- phi*={phi:.3f}, achieved_mean={achieved:.4f}, mean_abs_error={mean_err:.4f}, max_abs_error={max_err:.4f}, seeds={n_seed}".format(
                phi=phi,
                achieved=achieved,
                mean_err=mean_abs_error,
                max_err=max_abs_error,
                n_seed=n_seed,
            )
        )
        if max_abs_error > tolerance:
            print(
                "Warning: phi*={phi:.3f} calibration exceeded tolerance {tol:.3f}; max abs error={max_err:.4f}".format(
                    phi=phi,
                    tol=float(tolerance),
                    max_err=max_abs_error,
                )
            )


def _write_result1_phi_sweep_figures(rows: List[Dict[str, Any]], output_dir: str) -> List[str]:
    if not rows:
        return []
    csv_path = os.path.join(output_dir, "results.csv")
    if not os.path.isfile(csv_path):
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    outputs: List[str] = []
    metric_paths, metric_warnings = plot_phi_sweep_fairness_lines(
        csv_path=csv_path,
        output_dir=output_dir,
        archetype=1,
        N=5000,
    )
    outputs.extend(metric_paths)
    for msg in metric_warnings:
        print(f"Warning: {msg}")

    heatmap_path = os.path.join(output_dir, "modal_graph_frequency_heatmap.png")
    path, heatmap_warnings = plot_modal_graph_frequency_heatmap(
        csv_path=csv_path,
        output_path=heatmap_path,
        archetype=1,
        N=5000,
    )
    outputs.append(path)
    for msg in heatmap_warnings:
        print(f"Warning: {msg}")
    return outputs


def _write_result1_paper_phi_sweep(
    rows: List[Dict[str, Any]],
    output_dir: str,
    metrics: List[str] | None = None,
) -> str | None:
    if not rows:
        return None
    csv_path = os.path.join(output_dir, "results.csv")
    if not os.path.isfile(csv_path):
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    if metrics is None:
        metrics = ["te", "pse", "auroc"]
    metric_suffix = "_".join(metrics)
    output_path = os.path.join(output_dir, f"phi_sweep_{metric_suffix}_ieee.pdf")
    path, paper_warnings = plot_phi_sweep_paper_row(
        csv_path=csv_path,
        output_path=output_path,
        archetype=1,
        N=5000,
        metrics=metrics,
    )
    for msg in paper_warnings:
        print(f"Warning: {msg}")
    return path


def _write_result1_selectivity_plots(rows: List[Dict[str, Any]], output_dir: str) -> List[str]:
    df = pd.DataFrame(rows)
    outputs: List[str] = []
    allowed_algorithms = {"vanilla_ges", "hard_constraints", "fair_mec_phase1", "fair_mec_full"}
    for phi in [0.1, 0.5, 0.9]:
        phi_df = df[(df["archetype"] == 1) & (df["N"] == 5000) & (df["phi_target"].round(3) == round(phi, 3))]
        phi_df = phi_df[phi_df["algorithm"].isin(allowed_algorithms)]
        if phi_df.empty:
            continue
        csv_path = os.path.join(output_dir, f"result1_phi_{phi}.csv")
        phi_df.to_csv(csv_path, index=False)
        # Selectivity curve (TPR_fair vs TPR_legit)
        sel_path = os.path.join(output_dir, f"selectivity_phi_{phi}.png")
        plot_selectivity_curve(csv_path=csv_path, output_path=sel_path, phi=phi)
        outputs.append(sel_path)
    return outputs


def _dag_to_latex(dag_str: str) -> str:
    edges = str(dag_str).split("|")
    parts: List[str] = []
    for edge in edges:
        m = re.match(r"(\w+)->(\w+)", edge.strip())
        if m:
            parts.append(rf"{m.group(1)}{{\to}}{m.group(2)}")
        else:
            parts.append(edge.strip())
    return r",\, ".join(parts)


def _fmt_pm(mean: float, std: float, bold: bool = False) -> str:
    value = rf"{mean:.3f}{{\pm}}{std:.3f}"
    return rf"$\bm{{{value}}}$" if bold else rf"${value}$"


def _phi05_table_groups() -> List[Dict[str, Any]]:
    # Each entry may carry:
    #   bold            – render row in bold
    #   separator_before – insert \midrule before this row (within the fairness block)
    return [
        {"algorithm": "vanilla_ges",   "lambda_param": 0.0, "label": "Vanilla GES",  "lambda_label": "---", "bold": False, "separator_before": False},
        {"algorithm": "hard_constraints","lambda_param": 0.0, "label": "Hard Constr.", "lambda_label": "---", "bold": False, "separator_before": False},
        {"algorithm": "fair_mec_phase1", "lambda_param": RESULT1_SOFT_ONLY_LAMBDA, "label": "+Soft-only",   "lambda_label": f"{RESULT1_SOFT_ONLY_LAMBDA:g}", "bold": False, "separator_before": False},
        {"algorithm": "fair_mec_full", "lambda_param": 0.1, "label": "+ProxyFair",   "lambda_label": "0.1", "bold": False, "separator_before": False},
        {"algorithm": "fair_mec_full", "lambda_param": 0.5, "label": "+ProxyFair",   "lambda_label": "0.5", "bold": False, "separator_before": False},
        {"algorithm": "fair_mec_full", "lambda_param": 1.0, "label": "+ProxyFair",   "lambda_label": "1.0", "bold": False, "separator_before": False},
        {"algorithm": "fair_mec_full", "lambda_param": 2.0, "label": "+ProxyFair",   "lambda_label": "2.0", "bold": False, "separator_before": False},
    ]


def _build_result1_phi05_latex_table(rows: List[Dict[str, Any]]) -> str | None:
    df = pd.DataFrame(rows)
    required = {
        "phi_target",
        "algorithm",
        "lambda_param",
        "est_te",
        "est_ie_proxy",
        "dp_gap_lr",
        "auc_roc",
        "final_dag",
    }
    if df.empty or not required.issubset(df.columns):
        return None

    phi_df = df[df["phi_target"].round(3) == 0.5].copy()
    if phi_df.empty:
        return None

    # (spec, rendered_row_text) pairs — track spec alongside text for separator logic.
    table_entries: List[tuple[Dict[str, Any], str]] = []
    for spec in _phi05_table_groups():
        grp = phi_df[
            (phi_df["algorithm"] == spec["algorithm"])
            & (phi_df["lambda_param"].round(4) == round(float(spec["lambda_param"]), 4))
        ]
        if grp.empty:
            continue

        # All effect columns report mean(|value|) across seeds — the average
        # fairness-violation magnitude.  Signed means cancel when the synthetic
        # DGP randomly flips which group is advantaged, giving misleadingly
        # small values; magnitudes are stable and comparable across algorithms.
        if "cf_te" in grp.columns:
            est_te_mean = float(grp["cf_te"].abs().mean())
            est_te_std = float(grp["cf_te"].abs().std(ddof=1))
        else:
            est_te_mean = float(grp["est_te"].abs().mean())
            est_te_std = float(grp["est_te"].abs().std(ddof=1))

        # PSE: proxy-mediated effect; use CPT-based cf_nie when available
        if "cf_nie" in grp.columns:
            pse_mean = float(grp["cf_nie"].abs().mean())
            pse_std = float(grp["cf_nie"].abs().std(ddof=1))
        else:
            pse_mean = float(grp["est_ie_proxy"].abs().mean())
            pse_std = float(grp["est_ie_proxy"].abs().std(ddof=1))

        # NDE: non-proxy direct effect; use CPT-based cf_nde when available
        if "cf_nde" in grp.columns:
            nde_mean = float(grp["cf_nde"].abs().mean())
            nde_std = float(grp["cf_nde"].abs().std(ddof=1))
        else:
            nde_series = (grp["est_te"] - grp["est_ie_proxy"]).abs()
            nde_mean = float(nde_series.mean())
            nde_std = float(nde_series.std(ddof=1))

        dpd_mean = float(grp["dp_gap_lr"].abs().mean())
        dpd_std = float(grp["dp_gap_lr"].abs().std(ddof=1))

        # EO: equalized-odds gap. Prefer explicit column when present; otherwise
        # derive per seed as max(|TPR gap|, |FPR gap|).
        if "eo_gap" in grp.columns:
            eo_series = grp["eo_gap"].abs()
        elif {"tpr_gap", "fpr_gap"}.issubset(grp.columns):
            eo_series = grp[["tpr_gap", "fpr_gap"]].abs().max(axis=1)
        else:
            eo_series = pd.Series(np.nan, index=grp.index)
        eo_mean = float(eo_series.mean())
        eo_std = float(eo_series.std(ddof=1))

        auc_mean = float(grp["auc_roc"].abs().mean())
        auc_std = float(grp["auc_roc"].abs().std(ddof=1))

        dag_counts = grp["final_dag"].value_counts()
        modal_dag = str(dag_counts.index[0])
        modal_pct = 100.0 * float(dag_counts.iloc[0]) / float(len(grp))
        graph_cell = rf"${_dag_to_latex(modal_dag)}$\;\;({modal_pct:.0f}\%)"

        is_bold = spec.get("bold", False)
        label = str(spec["label"])
        if is_bold:
            label = rf"\textbf{{{label}}}"
            graph_cell = rf"\textbf{{{graph_cell}}}"

        row_text = "\n".join(
            [
                f"{label}  ({spec['lambda_label']})",
                f"  & {_fmt_pm(est_te_mean, est_te_std, bold=is_bold)}",
                f"  & {_fmt_pm(pse_mean, pse_std, bold=is_bold)}",
                f"  & {_fmt_pm(nde_mean, nde_std, bold=is_bold)}",
                f"  & {_fmt_pm(dpd_mean, dpd_std, bold=is_bold)}",
                f"  & {_fmt_pm(eo_mean, eo_std, bold=is_bold)}",
                f"  & {_fmt_pm(auc_mean, auc_std, bold=is_bold)}",
                f"  & {graph_cell} \\\\",
            ]
        )
        table_entries.append((spec, row_text))

    if not table_entries:
        return None

    # Build the table body, inserting \midrule between the baseline block and
    # the fairness block, and between lambda-pair groups (separator_before=True).
    body: List[str] = []
    in_fairness_block = False
    for spec, row_text in table_entries:
        is_baseline = spec["lambda_label"] == "---"
        if not is_baseline and not in_fairness_block:
            body.append(r"\midrule")
            in_fairness_block = True
        elif not is_baseline and spec.get("separator_before"):
            body.append(r"\midrule")
        body.append(row_text)

    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{%",
            r"Synthetic results at $\phi^* = 0.5$, $N = 5{,}000$,",
            r"over 50 seeds. Mean $\pm$ SD reported. Wilcoxon signed-rank tests",
            r"(Bonferroni-corrected): ProxyFair at $\lambda_f \!\in\! \{1.0, 2.0\}$",
            r"differs significantly",
            r"from Vanilla GES on all metrics ($p < 10^{-7}$).",
            r"}",
            r"\label{tab:main_results}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}l c c c c c c c@{}}",
            r"\toprule",
            r"\textbf{Algorithm ($\lambda_f$)}",
            r"  & \textbf{$|\text{TE}|$} $\downarrow$",
            r"  & \textbf{$|\text{PSE}|$} $\downarrow$",
            r"  & \textbf{$|\text{NDE}|$} $\downarrow$",
            r"  & \textbf{$|\text{DPD}|$} $\downarrow$",
            r"  & \textbf{$|\text{EO}|$} $\downarrow$",
            r"  & \textbf{AUROC} $\uparrow$",
            r"  & \textbf{Modal graph (freq.)} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _write_result1_phi05_table(rows: List[Dict[str, Any]], output_dir: str) -> str | None:
    latex = _build_result1_phi05_latex_table(rows)
    if not latex:
        return None
    out_path = os.path.join(output_dir, "result1_phi05_main_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
        f.write("\n")
    return out_path


def _print_displacement_calibration(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    required = {
        "calibration_corr_s1_s2_continuous",
        "calibration_r2_v1_s1_continuous",
        "calibration_r2_v2_s2_continuous",
        "calibration_nmi_v1_s1_discrete",
        "calibration_nmi_v2_s2_discrete",
        "calibration_cramers_v_v1_s1_discrete",
        "calibration_cramers_v_v2_s2_discrete",
    }
    if df.empty or not required.issubset(df.columns):
        return
    dedup = df.drop_duplicates(subset=["seed"]).copy()
    print("Displacement calibration diagnostics:")
    for col, label in [
        ("calibration_corr_s1_s2_continuous", "corr(S1,S2) continuous"),
        ("calibration_r2_v1_s1_continuous", "R2(V1~S1) continuous"),
        ("calibration_r2_v2_s2_continuous", "R2(V2~S2) continuous"),
        ("calibration_nmi_v1_s1_discrete", "NMI(V1,S1) discrete"),
        ("calibration_nmi_v2_s2_discrete", "NMI(V2,S2) discrete"),
        ("calibration_cramers_v_v1_s1_discrete", "CramersV(V1,S1) discrete"),
        ("calibration_cramers_v_v2_s2_discrete", "CramersV(V2,S2) discrete"),
    ]:
        print(
            "- {label}: mean={mean:.4f}, sd={sd:.4f}".format(
                label=label,
                mean=float(dedup[col].mean()),
                sd=float(dedup[col].std(ddof=1)) if len(dedup) > 1 else 0.0,
            )
        )


def _build_displacement_latex_table(rows: List[Dict[str, Any]]) -> str | None:
    df = pd.DataFrame(rows)
    required = {"condition", "abs_pse_s1", "abs_pse_s2", "abs_pse_total", "retained_z_y", "final_dag"}
    if df.empty or not required.issubset(df.columns):
        return None

    specs = [
        ("baseline", "Baseline"),
        ("penalize_s1", r"Penalize $\{S_1\}$"),
        ("penalize_s2", r"Penalize $\{S_2\}$"),
        ("penalize_joint", r"Penalize $\{S_1,S_2\}$"),
    ]
    body: List[str] = []
    for condition, label in specs:
        grp = df[df["condition"] == condition]
        if grp.empty:
            continue
        dag_counts = grp["final_dag"].value_counts()
        modal_dag = str(dag_counts.index[0]) if not dag_counts.empty else ""
        modal_pct = 100.0 * float(dag_counts.iloc[0]) / float(len(grp)) if not dag_counts.empty else 0.0
        graph_cell = rf"${_dag_to_latex(modal_dag)}$\;\;({modal_pct:.0f}\%)" if modal_dag else "---"
        body.append(
            "\n".join(
                [
                    label,
                    f"  & {_fmt_pm(float(grp['abs_pse_s1'].mean()), float(grp['abs_pse_s1'].std(ddof=1)))}",
                    f"  & {_fmt_pm(float(grp['abs_pse_s2'].mean()), float(grp['abs_pse_s2'].std(ddof=1)))}",
                    f"  & {_fmt_pm(float(grp['abs_pse_total'].mean()), float(grp['abs_pse_total'].std(ddof=1)))}",
                    f"  & ${float(grp['retained_z_y'].mean()):.2f}$",
                    f"  & {graph_cell} \\\\",
                ]
            )
        )

    if not body:
        return None

    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Multi-attribute synthetic displacement experiment, $N=5{,}000$, over 50 seeds. Mean $\pm$ SD reported for path-specific effects.}",
            r"\label{tab:synthetic_displacement}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{@{}l c c c c c@{}}",
            r"\toprule",
            r"\textbf{Condition}",
            r"  & \textbf{$|\mathrm{PSE}_{S_1}|$} $\downarrow$",
            r"  & \textbf{$|\mathrm{PSE}_{S_2}|$} $\downarrow$",
            r"  & \textbf{$|\mathrm{PSE}_{\mathrm{total}}|$} $\downarrow$",
            r"  & \textbf{$\Pr[Z{\to}Y]$} $\uparrow$",
            r"  & \textbf{Modal graph (freq.)} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def _write_displacement_table(rows: List[Dict[str, Any]], output_dir: str) -> str | None:
    latex = _build_displacement_latex_table(rows)
    if not latex:
        return None
    out_path = os.path.join(output_dir, "displacement_main_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
        f.write("\n")
    return out_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified synthetic Fair-MEC experiments (GES only).")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke grid")
    parser.add_argument("--n-jobs", type=int, default=-1, help="joblib parallel workers")
    parser.add_argument(
        "--mode",
        type=str,
        default="main_grid",
        choices=["main_grid", "figure1", "scenario_a", "scenario_b", "ablation_ges", "result1", "displacement"],
        help="Synthetic experiment mode to run",
    )
    parser.add_argument(
        "--plot-tradeoff",
        action="store_true",
        help="When mode=result1, also write per-phi selectivity CSVs and PNGs",
    )
    parser.add_argument(
        "--disable-temporal-blacklist",
        action="store_true",
        help="Disable temporal blacklisting in synthetic experiments.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of discretization bins for synthetic generation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-seed calibration diagnostics.")
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Skip running experiments and regenerate outputs from an existing results CSV. "
            "The CSV must live inside a result directory (its parent is used as output_dir). "
            "Only table/plot generation steps run; no experiments are executed."
        ),
    )
    parser.add_argument(
        "--paper-phi-sweep",
        action="store_true",
        help="When mode=result1, write the compact IEEE phi-sweep PDF.",
    )
    parser.add_argument(
        "--paper-phi-sweep-metrics",
        type=str,
        default="te,pse,auroc",
        help="Comma-separated metrics for --paper-phi-sweep (default: te,pse,auroc).",
    )
    args = parser.parse_args()
    _suppress_known_third_party_warnings()
    paper_phi_metrics = [
        metric.strip().lower()
        for metric in args.paper_phi_sweep_metrics.split(",")
        if metric.strip()
    ]

    if args.from_csv:
        csv_path = os.path.abspath(args.from_csv)
        if not os.path.isfile(csv_path):
            print(f"Error: --from-csv path does not exist: {csv_path}")
            raise SystemExit(1)
        out_dir = os.path.dirname(csv_path)
        rows = pd.read_csv(csv_path).to_dict("records")
        print(f"Loaded {len(rows)} rows from {csv_path}")
        if args.mode == "result1" and args.paper_phi_sweep:
            paper_path = _write_result1_paper_phi_sweep(
                rows=rows,
                output_dir=out_dir,
                metrics=paper_phi_metrics,
            )
            if paper_path:
                print(f"Generated IEEE phi-sweep PDF: {paper_path}")
        if args.mode == "result1" and args.plot_tradeoff:
            figures = _write_result1_selectivity_plots(rows=rows, output_dir=out_dir)
            if figures:
                print("Generated selectivity figures:")
                for fig in figures:
                    print(f"- {fig}")
        if args.mode == "result1":
            _print_phi_star_cramers_validation(rows)
            table_path = _write_result1_phi05_table(rows=rows, output_dir=out_dir)
            if table_path:
                print(f"Generated phi=0.5 LaTeX table: {table_path}")
            sweep_figures = _write_result1_phi_sweep_figures(rows=rows, output_dir=out_dir)
            if sweep_figures:
                print("Generated phi-sweep figures:")
                for fig in sweep_figures:
                    print(f"- {fig}")
        if args.mode == "displacement":
            _print_displacement_calibration(rows)
            table_path = _write_displacement_table(rows=rows, output_dir=out_dir)
            if table_path:
                print(f"Generated displacement LaTeX table: {table_path}")
        return

    enable_temporal_blacklist = not args.disable_temporal_blacklist
    tasks = build_tasks(
        smoke=args.smoke,
        mode=args.mode,
        enable_temporal_blacklist=enable_temporal_blacklist,
        n_bins=args.n_bins,
    )
    print(f"Running {len(tasks)} tasks with n_jobs={args.n_jobs}...")
    try:
        rows = _run_rows(tasks, n_jobs=args.n_jobs)
    except KeyboardInterrupt:
        print("Interrupted by user. No results were saved for this run.")
        raise SystemExit(130)
    _attach_task_metadata(rows, tasks)

    if args.smoke:
        _print_phi_summary(rows)
    if args.verbose:
        _print_phi_per_seed(rows)
    if args.mode == "result1":
        _print_phi_star_cramers_validation(rows)
    if args.mode == "displacement":
        _print_displacement_calibration(rows)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", "synthetic", f"{args.mode}_{timestamp}")
    save_results(
        rows,
        out_dir,
        config_snapshot={
            "mode": args.mode,
            "smoke": args.smoke,
            "n_jobs": args.n_jobs,
            "n_tasks": len(tasks),
            "plot_tradeoff": args.plot_tradeoff,
            "enable_temporal_blacklist": enable_temporal_blacklist,
            "n_bins": args.n_bins,
            "verbose": args.verbose,
        },
    )
    if args.mode == "result1" and args.plot_tradeoff:
        figures = _write_result1_selectivity_plots(rows=rows, output_dir=out_dir)
        if figures:
            print("Generated selectivity figures:")
            for fig in figures:
                print(f"- {fig}")
    if args.mode == "result1":
        table_path = _write_result1_phi05_table(rows=rows, output_dir=out_dir)
        if table_path:
            print("Generated phi=0.5 LaTeX table:")
            print(f"- {table_path}")
        sweep_figures = _write_result1_phi_sweep_figures(rows=rows, output_dir=out_dir)
        if sweep_figures:
            print("Generated phi-sweep figures:")
            for fig in sweep_figures:
                print(f"- {fig}")
        if args.paper_phi_sweep:
            paper_path = _write_result1_paper_phi_sweep(
                rows=rows,
                output_dir=out_dir,
                metrics=paper_phi_metrics,
            )
            if paper_path:
                print("Generated IEEE phi-sweep PDF:")
                print(f"- {paper_path}")
    if args.mode == "displacement":
        table_path = _write_displacement_table(rows=rows, output_dir=out_dir)
        if table_path:
            print("Generated displacement LaTeX table:")
            print(f"- {table_path}")
    print(f"Saved {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
