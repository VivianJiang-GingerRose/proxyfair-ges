"""Run the GESPhase2Only baseline from stored vanilla GES CPDAGs.

This baseline loads the cached vanilla GES CPDAG for each real dataset,
enumerates its consistent DAG extensions, selects the extension with minimum
Phase-2 fairness penalty, and evaluates that single selected DAG with the
standard downstream causal/ML pipeline.

Example:
    poetry run python src/experiments/run_ges_phase2_only.py --datasets law compas dutch bank
    poetry run python src/experiments/run_ges_phase2_only.py --dataset compas
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.experiments.compile_causal_fairness_analysis_table import _find_latest_run_dir  # noqa: E402
from src.experiments.main_runner import ExperimentConfig, FairCausalDiscoveryRunner, TeeLogger  # noqa: E402
from src.experiments.run_posthoc_pruning import (  # noqa: E402
    _build_general_graph,
    _enumerate_consistent_dags,
    _fit_scm_for_categorical_data_with_isolated_nodes,
    _load_baseline_config_payload,
    _load_cpdag_edges,
)
from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint  # noqa: E402
from src.faircausal.causal_learn.causallearn.graph.GeneralGraph import GeneralGraph  # noqa: E402
from src.faircausal.causal_learn.causallearn.graph.GraphNode import GraphNode  # noqa: E402
from src.faircausal.core import counterfactual_fairness_runner as cfr  # noqa: E402
from src.faircausal.core.fairness_scoring import (  # noqa: E402
    compute_phase2_penalty,
    compute_proxy_load,
    compute_proxy_strength_matrix,
)


DEFAULT_DATASETS = ["law", "compas", "dutch", "bank"]
DEFAULT_MAX_EXACT_EXTENSIONS = 1000
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ScoredDag:
    """A DAG extension and its Phase-2 fairness score."""

    ordinal: int
    dag_edges: List[Tuple[str, str]]
    psi_total: float


@contextlib.contextmanager
def _use_explicit_single_dag(dag_edges: Sequence[Tuple[str, str]]):
    """Force the shared evaluator to analyze one explicit DAG."""
    original_cpdag_to_dags = cfr.cpdag_to_dags
    original_fit_scm = cfr.fit_scm_for_categorical_data
    cfr.cpdag_to_dags = lambda _cpdag, node_names=None: [list(dag_edges)]
    cfr.fit_scm_for_categorical_data = _fit_scm_for_categorical_data_with_isolated_nodes
    try:
        yield
    finally:
        cfr.cpdag_to_dags = original_cpdag_to_dags
        cfr.fit_scm_for_categorical_data = original_fit_scm


def _dag_edges_to_general_graph(
    dag_edges: Sequence[Tuple[str, str]],
    node_names: Sequence[str],
) -> GeneralGraph:
    nodes = [GraphNode(name) for name in node_names]
    graph = GeneralGraph(nodes)
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    tail = Endpoint.TAIL.value
    arrow = Endpoint.ARROW.value

    for src, dst in dag_edges:
        if src not in name_to_idx or dst not in name_to_idx:
            continue
        si, ti = name_to_idx[src], name_to_idx[dst]
        graph.graph[si, ti] = tail
        graph.graph[ti, si] = arrow

    return graph


def _edge_string(dag_edges: Sequence[Tuple[str, str]]) -> str:
    return "|".join(f"{src}->{dst}" for src, dst in dag_edges)


def _compute_proxy_load(
    data: np.ndarray,
    node_names: List[str],
    protected_attr_indices: List[int],
) -> Dict[int, float]:
    proxy_strengths = compute_proxy_strength_matrix(data, node_names, protected_attr_indices)
    return compute_proxy_load(proxy_strengths, len(node_names), protected_attr_indices)


def _score_dags_by_phase2(
    dags: Sequence[List[Tuple[str, str]]],
    *,
    data: np.ndarray,
    node_names: List[str],
    proxy_load: Dict[int, float],
    outcome_index: int,
    beta: float = 0.0,
) -> List[ScoredDag]:
    scored: List[ScoredDag] = []
    for ordinal, dag_edges in enumerate(dags):
        dag_graph = _dag_edges_to_general_graph(dag_edges, node_names)
        psi_total = compute_phase2_penalty(
            dag_graph.graph,
            data,
            proxy_load,
            outcome_index,
            beta=beta,
        )
        scored.append(ScoredDag(ordinal=ordinal, dag_edges=list(dag_edges), psi_total=float(psi_total)))
    return scored


def _summarize_scored_dags(
    scored: Sequence[ScoredDag],
    *,
    random_seed: int,
    selection_mode: str,
) -> Dict[str, Any]:
    if not scored:
        raise ValueError("No scored DAGs available for selection")

    # Deterministic tie-break: earlier enumerator order wins.
    selected = min(scored, key=lambda row: (row.psi_total, row.ordinal))
    random_row = scored[0] if len(scored) == 1 else random.Random(random_seed).choice(list(scored))
    psi_values = [row.psi_total for row in scored]

    return {
        "selected": selected,
        "random": random_row,
        "n_extensions": int(len(scored)),
        "psi_total_selected": float(selected.psi_total),
        "psi_total_mean": float(np.mean(psi_values)),
        "psi_total_max": float(np.max(psi_values)),
        "psi_total_random": float(random_row.psi_total),
        "selected_dag_edges": _edge_string(selected.dag_edges),
        "random_dag_edges": _edge_string(random_row.dag_edges),
        "selection_mode": selection_mode,
    }


def _count_undirected_edges(cpdag: GeneralGraph) -> int:
    tail = Endpoint.TAIL.value
    n_nodes = len(cpdag.get_nodes())
    count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if cpdag.graph[i, j] == tail and cpdag.graph[j, i] == tail:
                count += 1
    return count


def _apply_meek_closure(graph: GeneralGraph, node_names: Sequence[str]) -> GeneralGraph:
    """Apply causal-learn Meek rules when available."""
    causal_learn_path = PROJECT_ROOT / "src" / "faircausal" / "causal_learn"
    if str(causal_learn_path) not in sys.path:
        sys.path.append(str(causal_learn_path))

    try:
        from causallearn.graph.GraphClass import CausalGraph
        from causallearn.utils.PCUtils.Meek import meek
    except Exception:
        return graph

    cg = CausalGraph(len(node_names), list(node_names))
    cg.G = graph
    with contextlib.redirect_stdout(io.StringIO()):
        return meek(cg).G


def _complete_partial_graph_for_scoring(graph: GeneralGraph) -> GeneralGraph:
    """Complete a PDAG with existing causal-learn utilities for fallback scoring."""
    causal_learn_path = PROJECT_ROOT / "src" / "faircausal" / "causal_learn"
    if str(causal_learn_path) not in sys.path:
        sys.path.append(str(causal_learn_path))

    from src.faircausal.causal_learn.causallearn.utils.PDAG2DAG import pdag2dag

    return pdag2dag(graph)


def _remaining_undirected_pairs(graph: GeneralGraph) -> List[Tuple[int, int]]:
    tail = Endpoint.TAIL.value
    pairs: List[Tuple[int, int]] = []
    for i in range(len(graph.get_nodes())):
        for j in range(i + 1, len(graph.get_nodes())):
            if graph.graph[i, j] == tail and graph.graph[j, i] == tail:
                pairs.append((i, j))
    return pairs


def _directed_edges_from_graph(graph: GeneralGraph, node_names: Sequence[str]) -> List[Tuple[str, str]]:
    tail = Endpoint.TAIL.value
    arrow = Endpoint.ARROW.value
    edges: List[Tuple[str, str]] = []
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i == j:
                continue
            if graph.graph[i, j] == tail and graph.graph[j, i] == arrow:
                edges.append((node_names[i], node_names[j]))
    return edges


def _orient_pair(graph: GeneralGraph, src_idx: int, dst_idx: int) -> GeneralGraph:
    import copy

    candidate = copy.deepcopy(graph)
    tail = Endpoint.TAIL.value
    arrow = Endpoint.ARROW.value
    candidate.graph[src_idx, dst_idx] = tail
    candidate.graph[dst_idx, src_idx] = arrow
    return candidate


def _greedy_phase2_orientation(
    cpdag: GeneralGraph,
    *,
    data: np.ndarray,
    node_names: List[str],
    proxy_load: Dict[int, float],
    outcome_index: int,
    beta: float,
) -> ScoredDag:
    """Defensive fallback for CPDAGs too large for exact extension enumeration."""
    graph = cpdag

    while _remaining_undirected_pairs(graph):
        best_graph: Optional[GeneralGraph] = None
        best_score = float("inf")

        for left, right in _remaining_undirected_pairs(graph):
            for src, dst in ((left, right), (right, left)):
                candidate = _orient_pair(graph, src, dst)
                candidate = _apply_meek_closure(candidate, node_names)
                try:
                    completed = _complete_partial_graph_for_scoring(candidate)
                except Exception:
                    continue
                psi_total = compute_phase2_penalty(
                    completed.graph,
                    data,
                    proxy_load,
                    outcome_index,
                    beta=beta,
                )
                if psi_total < best_score:
                    best_score = float(psi_total)
                    best_graph = candidate

        if best_graph is None:
            completed = _complete_partial_graph_for_scoring(graph)
            edges = _directed_edges_from_graph(completed, node_names)
            psi_total = compute_phase2_penalty(completed.graph, data, proxy_load, outcome_index, beta=beta)
            return ScoredDag(ordinal=0, dag_edges=edges, psi_total=float(psi_total))

        graph = best_graph

    edges = _directed_edges_from_graph(graph, node_names)
    psi_total = compute_phase2_penalty(graph.graph, data, proxy_load, outcome_index, beta=beta)
    return ScoredDag(ordinal=0, dag_edges=edges, psi_total=float(psi_total))


def _run_selected_dag_evaluation(
    *,
    dataset: str,
    runner: FairCausalDiscoveryRunner,
    df_scm_evaluation: pd.DataFrame,
    node_names: List[str],
    source_cpdag: GeneralGraph,
    selected_dag: List[Tuple[str, str]],
    baseline_run_dir: Path,
    cpdag_source: str,
    metadata: Dict[str, Any],
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
        fh.write("=== GESPHASE2ONLY EXPERIMENT LOG ===\n")
        fh.write(f"Dataset: {dataset}\n")
        fh.write(f"Baseline run dir: {baseline_run_dir}\n")
        fh.write(f"Baseline CPDAG source: {cpdag_source}\n")
        fh.write(f"n_extensions: {metadata['n_extensions']}\n")
        fh.write(f"selection_mode: {metadata['selection_mode']}\n")
        fh.write("Tie-breaking: deterministic enumerator order for equal Psi_Total.\n")
        fh.write(f"psi_total_selected: {metadata['psi_total_selected']}\n")
        fh.write(f"psi_total_mean: {metadata['psi_total_mean']}\n")
        fh.write(f"psi_total_max: {metadata['psi_total_max']}\n")
        fh.write(f"psi_total_random: {metadata['psi_total_random']}\n")
        fh.write(f"selected_dag_edges: {metadata['selected_dag_edges']}\n\n")

    original_stdout = sys.stdout
    tee_logger = TeeLogger(log_path, original_stdout)
    try:
        sys.stdout = tee_logger
        with _use_explicit_single_dag(selected_dag):
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
    cf_results_df.to_csv(results_dir / "ges_phase2_only_cf_results.csv", index=False)
    ml_results_df.to_csv(results_dir / "ges_phase2_only_ml_results.csv", index=False)

    config_payload = {
        "dataset_name": dataset,
        "experiment_type": "ges_phase2_only",
        "dataset_config": runner.dataset_config,
        "experiment_config": {
            "baseline_run_dir": str(baseline_run_dir),
            "baseline_cpdag_source": cpdag_source,
            "tie_breaking": "deterministic_enumerator_order",
            "phase2_beta": 0.0,
            "run_dir": str(output_run_dir),
            **{k: v for k, v in metadata.items() if k not in {"selected", "random"}},
        },
    }
    with open(results_dir / "ges_phase2_only_config.json", "w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, indent=2, default=str)

    return cf_results_df, ml_results_df


def run_dataset(
    *,
    dataset: str,
    results_root: Path,
    baseline_run_dir: Optional[Path],
    max_exact_extensions: int,
    random_seed: int,
    timestamp: str,
) -> Dict[str, Any]:
    dataset_root = results_root / dataset
    if baseline_run_dir is None:
        baseline_run_dir = _find_latest_run_dir(dataset_root, "baseline")
    baseline_run_dir = baseline_run_dir.resolve()

    config = ExperimentConfig(dataset_name=dataset, experiment_type="baseline")
    baseline_payload = _load_baseline_config_payload(baseline_run_dir)
    if baseline_payload and isinstance(baseline_payload.get("dataset_config"), dict):
        config.dataset_config = baseline_payload["dataset_config"]

    runner = FairCausalDiscoveryRunner(config)
    prep_run_dir = dataset_root / f"ges_phase2_only_prep_{timestamp}"
    prep_run_dir.mkdir(parents=True, exist_ok=True)
    (prep_run_dir / "scm_models").mkdir(exist_ok=True)
    (prep_run_dir / "plots").mkdir(exist_ok=True)
    runner.run_dir = prep_run_dir
    runner.log_file = prep_run_dir / "experiment_log.txt"
    with open(runner.log_file, "w", encoding="utf-8") as fh:
        fh.write("=== GESPHASE2ONLY PREP LOG ===\n")
        fh.write(f"Dataset: {dataset}\n")
        fh.write(f"Baseline run dir: {baseline_run_dir}\n")

    original_stdout = sys.stdout
    prep_logger = TeeLogger(runner.log_file, original_stdout)
    try:
        sys.stdout = prep_logger
        df_structure_discovery, df_scm_evaluation, _x, node_names = runner.load_and_split_data()
    finally:
        sys.stdout = original_stdout
        prep_logger.close()

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

    fairness_settings = runner.dataset_config["fairness_settings"]
    protected_indices = [node_names.index(attr) for attr in fairness_settings["protected_attrs"] if attr in node_names]
    outcome_index = node_names.index(fairness_settings["target"])
    beta = 0.0
    proxy_load = _compute_proxy_load(df_structure_discovery.values, node_names, protected_indices)

    n_undirected = _count_undirected_edges(source_cpdag)
    n_possible_orientations = 2 ** n_undirected

    if n_possible_orientations > int(max_exact_extensions):
        print(
            f"Warning: {dataset} baseline CPDAG has {n_undirected} undirected edges "
            f"({n_possible_orientations} orientations); using greedy fallback."
        )
        selected = _greedy_phase2_orientation(
            source_cpdag,
            data=df_structure_discovery.values,
            node_names=node_names,
            proxy_load=proxy_load,
            outcome_index=outcome_index,
            beta=beta,
        )
        metadata = {
            "selected": selected,
            "random": selected,
            "n_extensions": int(n_possible_orientations),
            "psi_total_selected": float(selected.psi_total),
            "psi_total_mean": float("nan"),
            "psi_total_max": float("nan"),
            "psi_total_random": float("nan"),
            "selected_dag_edges": _edge_string(selected.dag_edges),
            "random_dag_edges": "",
            "selection_mode": "greedy_fallback",
        }
    else:
        consistent_dags = _enumerate_consistent_dags(source_cpdag, node_names)
        scored = _score_dags_by_phase2(
            consistent_dags,
            data=df_structure_discovery.values,
            node_names=node_names,
            proxy_load=proxy_load,
            outcome_index=outcome_index,
            beta=beta,
        )
        selection_mode = "single_extension" if len(scored) == 1 else "exact"
        metadata = _summarize_scored_dags(
            scored,
            random_seed=random_seed,
            selection_mode=selection_mode,
        )

    output_run_dir = dataset_root / f"ges_phase2_only_{timestamp}"
    selected_dag = list(metadata["selected"].dag_edges)
    _run_selected_dag_evaluation(
        dataset=dataset,
        runner=runner,
        df_scm_evaluation=df_scm_evaluation,
        node_names=node_names,
        source_cpdag=source_cpdag,
        selected_dag=selected_dag,
        baseline_run_dir=baseline_run_dir,
        cpdag_source=cpdag_source,
        metadata=metadata,
        output_run_dir=output_run_dir,
    )

    summary_row = {
        "dataset": dataset,
        "baseline_run_dir": str(baseline_run_dir),
        "baseline_cpdag_source": cpdag_source,
        "n_extensions": metadata["n_extensions"],
        "psi_total_selected": metadata["psi_total_selected"],
        "psi_total_mean": metadata["psi_total_mean"],
        "psi_total_max": metadata["psi_total_max"],
        "psi_total_random": metadata["psi_total_random"],
        "selected_dag_edges": metadata["selected_dag_edges"],
        "random_dag_edges": metadata["random_dag_edges"],
        "selection_mode": metadata["selection_mode"],
    }
    return summary_row


def _write_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GESPhase2Only from existing vanilla GES CPDAGs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to run. Use one or more of: law compas dutch bank.",
    )
    parser.add_argument("--dataset", default=None, help="Single-dataset alias for --datasets.")
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
        "--max-exact-extensions",
        type=int,
        default=DEFAULT_MAX_EXACT_EXTENSIONS,
        help="Maximum orientation combinations to enumerate exactly before greedy fallback.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed for selecting the random audit DAG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [args.dataset] if args.dataset else list(args.datasets)
    if args.baseline_run_dir and len(datasets) != 1:
        raise ValueError("--baseline-run-dir can only be used with a single dataset")
    if args.max_exact_extensions <= 0:
        raise ValueError("--max-exact-extensions must be positive")

    results_root = Path(args.results_root)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        print(f"\n=== Running GESPhase2Only for {dataset} ===")
        row = run_dataset(
            dataset=dataset,
            results_root=results_root,
            baseline_run_dir=Path(args.baseline_run_dir) if args.baseline_run_dir else None,
            max_exact_extensions=int(args.max_exact_extensions),
            random_seed=int(args.random_seed),
            timestamp=timestamp,
        )
        summary_rows.append(row)
        print(
            f"Selected {dataset}: Psi_Total={row['psi_total_selected']:.6f}, "
            f"n_extensions={row['n_extensions']}, mode={row['selection_mode']}"
        )

    summary_dir = results_root / f"ges_phase2_only_summary_{timestamp}"
    _write_summary(summary_dir / "ges_phase2_only_summary.csv", summary_rows)
    print(f"\nWrote GESPhase2Only summary to: {summary_dir}")


if __name__ == "__main__":
    main()
