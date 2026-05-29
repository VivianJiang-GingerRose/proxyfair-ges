"""Run the post-hoc pruning baseline from stored baseline CPDAGs.

This script intentionally does not call GES. It loads the baseline CPDAG
already produced by an existing baseline run, enumerates its consistent DAGs
using the existing CPDAG enumerator, prunes each DAG for a tau_p sweep, and
passes the pruned DAGs through the existing evaluation pipeline.

Example:
    poetry run python src/experiments/run_posthoc_pruning.py --datasets compas
    poetry run python src/experiments/run_posthoc_pruning.py --datasets bank compas dutch law
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as _dt
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.experiments.compile_causal_fairness_analysis_table import (  # noqa: E402
    _extract_condition_metrics,
    _find_latest_run_dir,
    _infer_default_protected_attrs,
)
from src.experiments.main_runner import (  # noqa: E402
    ExperimentConfig,
    FairCausalDiscoveryRunner,
    TeeLogger,
    _format_lambda_for_run_name,
)
from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint  # noqa: E402
from src.faircausal.causal_learn.causallearn.graph.GeneralGraph import GeneralGraph  # noqa: E402
from src.faircausal.causal_learn.causallearn.graph.GraphNode import GraphNode  # noqa: E402
from src.faircausal.core import counterfactual_fairness_runner as cfr  # noqa: E402
from src.faircausal.core.fairness_scoring import (  # noqa: E402
    compute_directed_penalty,
    compute_edge_proxy_fractions,
    compute_outcome_relevance,
)


DEFAULT_DATASETS = ["bank", "compas", "dutch", "law"]
DEFAULT_TAU_GRID = np.linspace(0.01, 1.0, 20)


def _tau_token(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _load_baseline_config_payload(baseline_run_dir: Path) -> Optional[Dict[str, Any]]:
    results_dir = baseline_run_dir / "results"
    if not results_dir.exists():
        return None
    config_files = sorted(results_dir.glob("*_config.json"))
    if not config_files:
        return None
    return _read_json(config_files[0])


def _parse_edge_tuple_text(text: str) -> List[Tuple[str, str]]:
    """Parse a Python-list edge literal from old log lines when available."""
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    edges: List[Tuple[str, str]] = []
    for item in parsed:
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], str)
        ):
            edges.append((item[0], item[1]))
    return edges


def _parse_cpdag_from_experiment_log(log_path: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Extract the last printed baseline CPDAG block from experiment_log.txt."""
    if not log_path.exists():
        raise FileNotFoundError(f"Baseline log not found: {log_path}")

    directed_re = re.compile(r"^\s*\d+\.\s+(.+?)\s+-->\s+(.+?)\s*$")
    undirected_re = re.compile(r"^\s*\d+\.\s+(.+?)\s+--\s+(.+?)\s*$")
    dag_literal_re = re.compile(r"^DAG edges:\s*(\[.*\])\s*$")

    blocks: List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]] = []
    directed: List[Tuple[str, str]] = []
    undirected: List[Tuple[str, str]] = []
    in_block = False
    literal_dags: List[List[Tuple[str, str]]] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            literal_match = dag_literal_re.match(line)
            if literal_match:
                literal_edges = _parse_edge_tuple_text(literal_match.group(1))
                if literal_edges:
                    literal_dags.append(literal_edges)

            if "FINAL GES RESULT - DETAILED EDGE ANALYSIS" in line:
                if in_block and (directed or undirected):
                    blocks.append((directed, undirected))
                directed = []
                undirected = []
                in_block = True
                continue

            if not in_block:
                continue

            directed_match = directed_re.match(line)
            if directed_match:
                directed.append((directed_match.group(1).strip(), directed_match.group(2).strip()))
                continue

            undirected_match = undirected_re.match(line)
            if undirected_match and "-->" not in line:
                undirected.append((undirected_match.group(1).strip(), undirected_match.group(2).strip()))

    if in_block and (directed or undirected):
        blocks.append((directed, undirected))

    if blocks:
        # Old logs may contain later appended graph dumps from follow-up runs or
        # plotting/debug sessions. The first GES result block is the baseline
        # CPDAG produced by that run, and matches the existing baseline tables.
        return blocks[0]

    # Some logs only contain enumerated DAG edge literals. This is a fallback
    # for already-oriented baseline outputs.
    if literal_dags:
        return literal_dags[0], []

    raise ValueError(f"Could not parse baseline CPDAG edges from {log_path}")


def _load_cpdag_edges(
    baseline_run_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """Load baseline CPDAG edges from JSON if available, otherwise from logs."""
    candidate_jsons = [
        baseline_run_dir / "results" / "baseline_cpdag.json",
        baseline_run_dir / "baseline_cpdag.json",
        baseline_run_dir / "source_baseline_cpdag.json",
    ]
    for path in candidate_jsons:
        payload = _read_json(path)
        if not payload:
            continue
        directed = [tuple(edge) for edge in payload.get("directed_edges", [])]
        undirected = [tuple(edge) for edge in payload.get("undirected_edges", [])]
        return directed, undirected, str(path)

    directed, undirected = _parse_cpdag_from_experiment_log(baseline_run_dir / "experiment_log.txt")
    return directed, undirected, str(baseline_run_dir / "experiment_log.txt")


def _build_general_graph(
    node_names: Sequence[str],
    directed_edges: Sequence[Tuple[str, str]],
    undirected_edges: Sequence[Tuple[str, str]],
) -> GeneralGraph:
    nodes = [GraphNode(name) for name in node_names]
    graph = GeneralGraph(nodes)
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    tail = Endpoint.TAIL.value
    arrow = Endpoint.ARROW.value

    for src, dst in directed_edges:
        if src not in name_to_idx or dst not in name_to_idx:
            continue
        si, ti = name_to_idx[src], name_to_idx[dst]
        graph.graph[si, ti] = tail
        graph.graph[ti, si] = arrow

    for left, right in undirected_edges:
        if left not in name_to_idx or right not in name_to_idx:
            continue
        li, ri = name_to_idx[left], name_to_idx[right]
        graph.graph[li, ri] = tail
        graph.graph[ri, li] = tail

    return graph


def _prune_dag_edges(
    dag_edges: Sequence[Tuple[str, str]],
    *,
    tau_p: float,
    node_names: Sequence[str],
    pi_sigma: np.ndarray,
    rho_adj: Dict[int, float],
) -> List[Tuple[str, str]]:
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    pruned: List[Tuple[str, str]] = []
    for src, dst in dag_edges:
        if src not in name_to_idx or dst not in name_to_idx:
            continue
        penalty = compute_directed_penalty(
            pi_sigma,
            rho_adj,
            source_idx=name_to_idx[src],
            target_idx=name_to_idx[dst],
        )
        if penalty <= tau_p:
            pruned.append((src, dst))
    return pruned


def _enumerate_consistent_dags(cpdag: GeneralGraph, node_names: Sequence[str]) -> List[List[Tuple[str, str]]]:
    """Call the existing CPDAG enumerator without leaking encoding-sensitive debug output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return cfr.cpdag_to_dags(cpdag, list(node_names))


def _fit_scm_for_categorical_data_with_isolated_nodes(
    df: pd.DataFrame,
    dag: Sequence[Tuple[str, str]],
    var_types: Dict[str, str],
    random_state: int = 42,
):
    """SCM fit compatible with edge-list DAGs that may leave isolated nodes.

    This mirrors the existing fitter, but adds all dataframe columns as graph
    nodes before fitting. That keeps pruning from accidentally dropping a node
    just because all of its incident edges were removed.
    """
    _ = random_state
    try:
        print("Creating StructuralCausalModel for categorical data...")
        graph = nx.DiGraph()
        graph.add_nodes_from(df.columns)
        graph.add_edges_from(list(dag))
        causal_model = cfr.gcm.StructuralCausalModel(graph)

        df_categorical = df.copy()
        categorical_info: Dict[str, Dict[str, Any]] = {}
        for col in df_categorical.columns:
            if col in var_types and var_types[col] in ["categorical", "binary"]:
                unique_values = sorted(df_categorical[col].unique())
                int_to_string = {int(v): str(v) for v in unique_values}
                string_to_int = {str(v): int(v) for v in unique_values}
                df_categorical[col] = df_categorical[col].astype(str)
                categorical_info[col] = {
                    "original_int_values": unique_values,
                    "int_to_string": int_to_string,
                    "string_to_int": string_to_int,
                    "fallback_value": unique_values[0] if unique_values else 0,
                }
                print(f"Converted {col} ({var_types[col]}) to strings: {unique_values}")

        causal_model._categorical_info = categorical_info

        root_nodes = set(graph.nodes())
        for _src, dst in graph.edges():
            root_nodes.discard(dst)

        print(f"Root nodes: {root_nodes}")
        for node in graph.nodes():
            if node not in df_categorical.columns:
                continue
            if node in root_nodes:
                print(f"Setting EmpiricalDistribution for ROOT node: {node}")
                causal_model.set_causal_mechanism(node, cfr.gcm.EmpiricalDistribution())
            else:
                print(f"Assigning ClassifierFCM(RandomForest) to NON-ROOT node: {node}")
                causal_model.set_causal_mechanism(
                    node,
                    cfr.gcm.ClassifierFCM(classifier_model=cfr.create_random_forest_classifier()),
                )

        print("Fitting causal model...")
        df_scm = cfr.get_data_for_algorithm(df_categorical, "scm")
        cfr.gcm.fit(causal_model, df_scm)
        print("StructuralCausalModel fitted successfully for categorical data")
        return causal_model
    except Exception as exc:
        print(f"Error fitting causal model: {exc}")
        import traceback

        traceback.print_exc()
        return None


@contextlib.contextmanager
def _use_explicit_dag_list(dags: Sequence[List[Tuple[str, str]]]):
    original_cpdag_to_dags = cfr.cpdag_to_dags
    original_fit_scm = cfr.fit_scm_for_categorical_data
    cfr.cpdag_to_dags = lambda _cpdag, node_names=None: list(dags)
    cfr.fit_scm_for_categorical_data = _fit_scm_for_categorical_data_with_isolated_nodes
    try:
        yield
    finally:
        cfr.cpdag_to_dags = original_cpdag_to_dags
        cfr.fit_scm_for_categorical_data = original_fit_scm


def _mean_and_std(values: Iterable[float]) -> Tuple[float, float]:
    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _overall_condition_metrics(condition: Any, protected_attrs: Sequence[str]) -> Dict[str, float]:
    def mean_attr(key: str) -> float:
        return float(
            np.nanmean([
                condition.attr_effects.get(attr, {}).get(key, float("nan"))
                for attr in protected_attrs
            ])
        )

    def mean_attr_std(key: str) -> float:
        return float(
            np.nanmean([
                condition.attr_effects_std.get(attr, {}).get(key, float("nan"))
                for attr in protected_attrs
            ])
        )

    return {
        "overall_te_abs": mean_attr("te_abs"),
        "overall_te_abs_std": mean_attr_std("te_abs"),
        "overall_nde_abs": mean_attr("nde_abs"),
        "overall_nde_abs_std": mean_attr_std("nde_abs"),
        "overall_pse_abs": mean_attr("pse_abs"),
        "overall_pse_abs_std": mean_attr_std("pse_abs"),
        "overall_ite_abs": mean_attr("ite_abs"),
        "overall_ite_abs_std": mean_attr_std("ite_abs"),
        "overall_dpd": float(np.nanmean([condition.dpd_per_attr.get(a, float("nan")) for a in protected_attrs])),
        "overall_dpd_std": float(np.nanmean([condition.dpd_std_per_attr.get(a, float("nan")) for a in protected_attrs])),
        "overall_eo": float(np.nanmean([condition.eo_per_attr.get(a, float("nan")) for a in protected_attrs])),
        "overall_eo_std": float(np.nanmean([condition.eo_std_per_attr.get(a, float("nan")) for a in protected_attrs])),
        "f1": float(condition.f1),
        "f1_std": float(condition.f1_std),
        "auroc": float(condition.auroc),
        "auroc_std": float(condition.auroc_std),
    }


def _select_best_tau(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No tau rows available for selection")
    higher_is_better = metric in {"auroc", "f1"}
    finite = [row for row in rows if metric in row and not math.isnan(float(row[metric]))]
    if not finite:
        return rows[0]
    return max(finite, key=lambda row: float(row[metric])) if higher_is_better else min(finite, key=lambda row: float(row[metric]))


def _format_pm(mean: float, std: float, decimals: int = 4) -> str:
    if mean is None or math.isnan(float(mean)):
        return "N/A"
    if std is None or math.isnan(float(std)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _render_best_tau_latex(best_rows: Sequence[Dict[str, Any]], selection_metric: str) -> str:
    selection_label = selection_metric.replace("_", r"\_")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Post-hoc pruning baseline at the best $\tau_p$ per dataset. "
        + f"Best is selected by {selection_label}. "
        + r"Entries show mean $\pm$ std across pruned DAGs.}",
        r"\label{tab:posthoc_pruning_best_tau}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"Dataset & $\tau_p$ & \# DAGs & $|\mathrm{TE}|$ & $|\mathrm{NDE}|$ & $|\mathrm{PSE}|$ & DPD & EO & AUROC \\",
        r"\midrule",
    ]
    for row in best_rows:
        dataset = str(row["dataset"])
        lines.append(
            " & ".join(
                [
                    dataset.capitalize(),
                    f"{float(row['tau_p']):.4f}",
                    str(int(row["n_pruned_dags"])),
                    _format_pm(row["overall_te_abs"], row["overall_te_abs_std"]),
                    _format_pm(row["overall_nde_abs"], row["overall_nde_abs_std"]),
                    _format_pm(row["overall_pse_abs"], row["overall_pse_abs_std"]),
                    _format_pm(row["overall_dpd"], row["overall_dpd_std"]),
                    _format_pm(row["overall_eo"], row["overall_eo_std"]),
                    _format_pm(row["auroc"], row["auroc_std"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_tau_evaluation(
    *,
    dataset: str,
    runner: FairCausalDiscoveryRunner,
    df_scm_evaluation: pd.DataFrame,
    node_names: List[str],
    source_cpdag: GeneralGraph,
    pruned_dags: List[List[Tuple[str, str]]],
    tau_p: float,
    tau_grid: Sequence[float],
    baseline_run_dir: Path,
    cpdag_source: str,
    n_consistent_dags: int,
    output_run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fairness_settings = runner.dataset_config["fairness_settings"]
    variable_settings = runner.dataset_config["variable_settings"]
    experiment_settings = runner.dataset_config["experiment_settings"]
    counterfactual_settings = runner.dataset_config.get("counterfactual_settings", {})

    protected_attrs = fairness_settings["protected_attrs"]
    target = fairness_settings["target"]
    disadvantage_group = fairness_settings["disadvantage_group"]
    var_types = variable_settings["var_types"]
    x_cols = [col for col in node_names if col != target]

    sample_percentage = getattr(
        runner.config,
        "sample_percentage",
        counterfactual_settings.get("sample_percentage", 1.0),
    )
    min_samples = getattr(runner.config, "min_samples", counterfactual_settings.get("min_samples", 10))
    random_state = getattr(runner.config, "random_state", counterfactual_settings.get("random_state", 42))
    cutoff_threshold = getattr(
        runner.config,
        "cutoff_threshold",
        counterfactual_settings.get("cutoff_threshold", 0.5),
    )
    calculate_pse = getattr(
        runner.config,
        "calculate_pse",
        counterfactual_settings.get("calculate_pse", False),
    )
    pse_sample_size = getattr(
        runner.config,
        "pse_sample_size",
        counterfactual_settings.get("pse_sample_size", None),
    )
    max_paths = getattr(runner.config, "max_paths", counterfactual_settings.get("max_paths", None))

    generate_synthetic = experiment_settings.get("generate_synthetic", True)
    generate_synthetic_double = experiment_settings.get("synthetic_variants", {}).get("double", False)
    generate_synthetic_balanced = experiment_settings.get("synthetic_variants", {}).get("balanced", False)
    generate_synthetic_protected = experiment_settings.get("synthetic_variants", {}).get("protected", False)

    if calculate_pse and (generate_synthetic_double or generate_synthetic_balanced or generate_synthetic_protected):
        generate_synthetic_double = False
        generate_synthetic_balanced = False
        generate_synthetic_protected = False

    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "results").mkdir(exist_ok=True)
    (output_run_dir / "scm_models").mkdir(exist_ok=True)
    (output_run_dir / "plots").mkdir(exist_ok=True)

    log_path = output_run_dir / "experiment_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("=== POST-HOC PRUNING EXPERIMENT LOG ===\n")
        fh.write(f"Dataset: {dataset}\n")
        fh.write(f"tau_p: {tau_p}\n")
        fh.write(f"Baseline run dir: {baseline_run_dir}\n")
        fh.write(f"Baseline CPDAG source: {cpdag_source}\n")
        fh.write(f"Consistent DAGs loaded: {n_consistent_dags}\n")
        fh.write(f"Pruned DAGs evaluated: {len(pruned_dags)}\n\n")

    original_stdout = sys.stdout
    tee_logger = TeeLogger(log_path, original_stdout)
    try:
        sys.stdout = tee_logger
        with _use_explicit_dag_list(pruned_dags):
            cf_results_df, ml_results_df = cfr.analyze_all_dags_improved_flow(
                cpdag=source_cpdag,
                df=df_scm_evaluation,
                node_names=node_names,
                protected_attrs=protected_attrs,
                target=target,
                X_cols=x_cols,
                disadvantage_group=disadvantage_group,
                estimator=RandomForestClassifier,
                var_types=var_types,
                max_dags=None,
                log_file=log_path,
                save_model=experiment_settings.get("save_models", False),
                model_dir=output_run_dir / "scm_models",
                generate_synthetic=generate_synthetic,
                generate_synthetic_double=generate_synthetic_double,
                generate_synthetic_balanced=generate_synthetic_balanced,
                generate_synthetic_protected=generate_synthetic_protected,
                sample_percentage=sample_percentage,
                min_samples=min_samples,
                random_state=random_state,
                cutoff_threshold=cutoff_threshold,
                calculate_pse=calculate_pse,
                pse_sample_size=pse_sample_size,
                max_paths=max_paths,
            )
    finally:
        sys.stdout = original_stdout
        tee_logger.close()

    results_dir = output_run_dir / "results"
    cf_results_df.to_csv(results_dir / "posthoc_pruning_cf_results.csv", index=False)
    ml_results_df.to_csv(results_dir / "posthoc_pruning_ml_results.csv", index=False)

    config_payload = {
        "dataset_name": dataset,
        "experiment_type": "posthoc_pruning",
        "dataset_config": runner.dataset_config,
        "experiment_config": {
            "tau_p": float(tau_p),
            "tau_grid": [float(v) for v in tau_grid],
            "baseline_run_dir": str(baseline_run_dir),
            "baseline_cpdag_source": cpdag_source,
            "n_consistent_dags": int(n_consistent_dags),
            "n_pruned_dags": int(len(pruned_dags)),
            "run_dir": str(output_run_dir),
        },
    }
    with open(results_dir / "posthoc_pruning_config.json", "w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, indent=2, default=str)

    return cf_results_df, ml_results_df


def run_dataset(
    *,
    dataset: str,
    results_root: Path,
    baseline_run_dir: Optional[Path],
    tau_grid: Sequence[float],
    max_dags_override: Optional[int],
    all_dags: bool,
    selection_metric: str,
    timestamp: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset_root = results_root / dataset
    if baseline_run_dir is None:
        baseline_run_dir = _find_latest_run_dir(dataset_root, "baseline")
    baseline_run_dir = baseline_run_dir.resolve()

    config = ExperimentConfig(dataset_name=dataset, experiment_type="baseline")
    baseline_payload = _load_baseline_config_payload(baseline_run_dir)
    if baseline_payload and isinstance(baseline_payload.get("dataset_config"), dict):
        config.dataset_config = baseline_payload["dataset_config"]

    runner = FairCausalDiscoveryRunner(config)
    prep_run_dir = dataset_root / f"posthoc_pruning_prep_{timestamp}"
    prep_run_dir.mkdir(parents=True, exist_ok=True)
    (prep_run_dir / "scm_models").mkdir(exist_ok=True)
    (prep_run_dir / "plots").mkdir(exist_ok=True)
    runner.run_dir = prep_run_dir
    runner.log_file = prep_run_dir / "experiment_log.txt"
    with open(runner.log_file, "w", encoding="utf-8") as fh:
        fh.write("=== POST-HOC PRUNING PREP LOG ===\n")
        fh.write(f"Dataset: {dataset}\n")
        fh.write(f"Baseline run dir: {baseline_run_dir}\n")

    df_structure_discovery, df_scm_evaluation, _x, node_names = runner.load_and_split_data()

    directed_edges, undirected_edges, cpdag_source = _load_cpdag_edges(baseline_run_dir)
    source_cpdag = _build_general_graph(node_names, directed_edges, undirected_edges)
    with open(prep_run_dir / "source_baseline_cpdag.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "node_names": node_names,
                "directed_edges": list(map(list, directed_edges)),
                "undirected_edges": list(map(list, undirected_edges)),
                "source": cpdag_source,
            },
            fh,
            indent=2,
        )

    consistent_dags = _enumerate_consistent_dags(source_cpdag, node_names)
    n_consistent_dags = len(consistent_dags)
    if not all_dags:
        configured_max = runner.dataset_config.get("experiment_settings", {}).get("max_dags", 100)
        max_dags = max_dags_override if max_dags_override is not None else configured_max
        if max_dags and len(consistent_dags) > int(max_dags):
            consistent_dags = consistent_dags[: int(max_dags)]

    fairness_settings = runner.dataset_config["fairness_settings"]
    protected_indices = [node_names.index(attr) for attr in fairness_settings["protected_attrs"] if attr in node_names]
    outcome_index = node_names.index(fairness_settings["target"])
    tau_c = runner.dataset_config.get("soft_fairness_settings", {}).get("tau_c", 5.0)
    pi_sigma = compute_edge_proxy_fractions(
        df_structure_discovery.values,
        protected_attr_indices=protected_indices,
        tau_c=tau_c,
        min_group_size=10,
        var_names=node_names,
    )
    rho_adj = compute_outcome_relevance(df_structure_discovery.values, outcome_index)

    _infer_primary, _infer_secondary, protected_attrs = _infer_default_protected_attrs(runner.dataset_config)
    protected_attrs = protected_attrs or fairness_settings["protected_attrs"]

    baseline_edge_counts = [len(dag) for dag in consistent_dags]
    baseline_edge_mean, baseline_edge_std = _mean_and_std(baseline_edge_counts)

    sweep_rows: List[Dict[str, Any]] = []
    for tau_p in tau_grid:
        token = _tau_token(float(tau_p))
        run_dir = dataset_root / f"posthoc_pruning_{timestamp}_tau_{token}"
        pruned_dags = [
            _prune_dag_edges(
                dag,
                tau_p=float(tau_p),
                node_names=node_names,
                pi_sigma=pi_sigma,
                rho_adj=rho_adj,
            )
            for dag in consistent_dags
        ]

        _run_tau_evaluation(
            dataset=dataset,
            runner=runner,
            df_scm_evaluation=df_scm_evaluation,
            node_names=node_names,
            source_cpdag=source_cpdag,
            pruned_dags=pruned_dags,
            tau_p=float(tau_p),
            tau_grid=tau_grid,
            baseline_run_dir=baseline_run_dir,
            cpdag_source=cpdag_source,
            n_consistent_dags=n_consistent_dags,
            output_run_dir=run_dir,
        )

        condition = _extract_condition_metrics(
            run_dir=run_dir,
            experiment="posthoc_pruning",
            protected_attrs=list(protected_attrs),
            primary_attr=protected_attrs[0],
        )
        overall = _overall_condition_metrics(condition, protected_attrs)
        edge_counts = [len(dag) for dag in pruned_dags]
        edge_mean, edge_std = _mean_and_std(edge_counts)
        pruned_edges_mean = float(baseline_edge_mean) - float(edge_mean)
        row = {
            "dataset": dataset,
            "tau_p": float(tau_p),
            "run_dir": str(run_dir),
            "baseline_run_dir": str(baseline_run_dir),
            "baseline_cpdag_source": cpdag_source,
            "n_consistent_dags": int(n_consistent_dags),
            "n_pruned_dags": int(len(pruned_dags)),
            "n_edges_before_mean": baseline_edge_mean,
            "n_edges_before_std": baseline_edge_std,
            "n_edges_mean": edge_mean,
            "n_edges_std": edge_std,
            "n_pruned_edges_mean": pruned_edges_mean,
            **overall,
        }
        sweep_rows.append(row)

    sweep_dir = dataset_root / f"posthoc_pruning_sweep_{timestamp}"
    _write_csv(sweep_dir / "posthoc_pruning_sweep_metrics.csv", sweep_rows)
    best = _select_best_tau(sweep_rows, selection_metric)
    _write_csv(sweep_dir / "posthoc_pruning_best_tau_summary.csv", [best])
    (sweep_dir / "posthoc_pruning_best_tau_table.tex").write_text(
        _render_best_tau_latex([best], selection_metric),
        encoding="utf-8",
    )
    return sweep_rows, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc pruning baseline from existing baseline CPDAGs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to run. Use one or more of: bank compas dutch law.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single-dataset alias for --datasets.",
    )
    parser.add_argument(
        "--results-root",
        default="results/experiments",
        help="Root folder containing experiment result directories.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        default=None,
        help="Explicit baseline run directory. Only valid with one dataset.",
    )
    parser.add_argument(
        "--tau-values",
        nargs="+",
        type=float,
        default=None,
        help="Space-separated tau_p values. Defaults to np.linspace(0.01, 1.0, 20).",
    )
    parser.add_argument(
        "--max-dags",
        type=int,
        default=None,
        help="Override number of baseline consistent DAGs to evaluate per tau.",
    )
    parser.add_argument(
        "--all-dags",
        action="store_true",
        help="Evaluate all enumerated DAGs instead of the dataset max_dags cap.",
    )
    parser.add_argument(
        "--selection-metric",
        default="overall_pse_abs",
        choices=[
            "overall_te_abs",
            "overall_nde_abs",
            "overall_pse_abs",
            "overall_dpd",
            "overall_eo",
            "auroc",
            "f1",
        ],
        help="Metric used to pick the best tau_p for the summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [args.dataset] if args.dataset else list(args.datasets)
    results_root = Path(args.results_root)
    if args.baseline_run_dir and len(datasets) != 1:
        raise ValueError("--baseline-run-dir can only be used with a single dataset")

    if args.tau_values is not None:
        tau_grid = list(args.tau_values)
    else:
        tau_grid = [float(v) for v in DEFAULT_TAU_GRID]

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_sweep_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        print(f"\n=== Running post-hoc pruning for {dataset} ===")
        sweep_rows, best = run_dataset(
            dataset=dataset,
            results_root=results_root,
            baseline_run_dir=Path(args.baseline_run_dir) if args.baseline_run_dir else None,
            tau_grid=tau_grid,
            max_dags_override=args.max_dags,
            all_dags=bool(args.all_dags),
            selection_metric=args.selection_metric,
            timestamp=timestamp,
        )
        all_sweep_rows.extend(sweep_rows)
        best_rows.append(best)
        print(
            f"Best {dataset}: tau_p={best['tau_p']:.4f}, "
            f"{args.selection_metric}={best[args.selection_metric]:.6f}"
        )

    summary_dir = results_root / f"posthoc_pruning_summary_{timestamp}"
    _write_csv(summary_dir / "posthoc_pruning_all_sweep_metrics.csv", all_sweep_rows)
    _write_csv(summary_dir / "posthoc_pruning_best_tau_summary.csv", best_rows)
    (summary_dir / "posthoc_pruning_best_tau_table.tex").write_text(
        _render_best_tau_latex(best_rows, args.selection_metric),
        encoding="utf-8",
    )
    print(f"\nWrote post-hoc pruning summary to: {summary_dir}")


if __name__ == "__main__":
    main()
