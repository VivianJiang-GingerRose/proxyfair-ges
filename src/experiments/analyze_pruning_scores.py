"""Analyze PostHocPruning edge scores from stored baseline CPDAGs.

This script is read-only with respect to experiments: it loads existing
baseline CPDAG artifacts, computes the same precomputed proxy inputs used by
PostHocPruning, and reports per-edge pruning scores for CPDAG skeleton edges.

Example:
    poetry run python src/experiments/analyze_pruning_scores.py --datasets law dutch bank
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.experiments.compile_causal_fairness_analysis_table import _find_latest_run_dir  # noqa: E402
from src.experiments.main_runner import ExperimentConfig, FairCausalDiscoveryRunner  # noqa: E402
from src.experiments.run_posthoc_pruning import (  # noqa: E402
    _build_general_graph,
    _load_baseline_config_payload,
    _load_cpdag_edges,
)
from src.faircausal.core.fairness_scoring import (  # noqa: E402
    compute_edge_proxy_fractions,
    compute_outcome_relevance,
)


DEFAULT_DATASETS = ["law", "dutch", "bank"]


def _ordered_skeleton_edges(
    directed_edges: Sequence[Tuple[str, str]],
    undirected_edges: Sequence[Tuple[str, str]],
    node_names: Sequence[str],
) -> List[Tuple[str, str]]:
    node_index = {name: idx for idx, name in enumerate(node_names)}
    seen: set[Tuple[str, str]] = set()

    for src, dst in list(directed_edges) + list(undirected_edges):
        if src not in node_index or dst not in node_index:
            raise ValueError(f"Edge ({src}, {dst}) includes variable not found in node_names")
        if src == dst:
            continue
        if node_index[src] <= node_index[dst]:
            key = (src, dst)
        else:
            key = (dst, src)
        seen.add(key)

    return sorted(seen, key=lambda pair: (node_index[pair[0]], node_index[pair[1]]))


def analyze_dataset(
    *,
    dataset: str,
    results_root: Path,
    baseline_run_dir: Optional[Path],
) -> pd.DataFrame:
    dataset_root = results_root / dataset
    if baseline_run_dir is None:
        baseline_run_dir = _find_latest_run_dir(dataset_root, "baseline")
    baseline_run_dir = baseline_run_dir.resolve()

    config = ExperimentConfig(dataset_name=dataset, experiment_type="baseline")
    baseline_payload = _load_baseline_config_payload(baseline_run_dir)
    if baseline_payload and isinstance(baseline_payload.get("dataset_config"), dict):
        config.dataset_config = baseline_payload["dataset_config"]

    with tempfile.TemporaryDirectory(prefix=f"pruning_score_analysis_{dataset}_") as tmpdir:
        runner = FairCausalDiscoveryRunner(config)
        runner.run_dir = Path(tmpdir)
        runner.log_file = runner.run_dir / "analysis_log.txt"
        with open(runner.log_file, "w", encoding="utf-8") as fh:
            fh.write("=== PRUNING SCORE ANALYSIS LOG ===\n")
            fh.write(f"Dataset: {dataset}\n")
            fh.write(f"Baseline run dir: {baseline_run_dir}\n")
            fh.write(f"Timestamp: {_dt.datetime.now().isoformat()}\n")

        df_structure_discovery, _df_scm_evaluation, _x, node_names = runner.load_and_split_data()
        if list(df_structure_discovery.columns) != node_names:
            raise ValueError(
                "Column-order mismatch between dataframe columns and node_names; "
                "cannot safely map CPDAG edge names to pi_sigma indices."
            )

        directed_edges, undirected_edges, cpdag_source = _load_cpdag_edges(baseline_run_dir)
        # Build once to ensure parity with baseline CPDAG parsing/validation logic.
        _ = _build_general_graph(node_names, directed_edges, undirected_edges)
        skeleton_edges = _ordered_skeleton_edges(directed_edges, undirected_edges, node_names)

        fairness_settings = runner.dataset_config["fairness_settings"]
        node_index = {name: idx for idx, name in enumerate(node_names)}

        protected_indices = [
            node_index[attr]
            for attr in fairness_settings["protected_attrs"]
            if attr in node_index
        ]
        outcome = fairness_settings["target"]
        if outcome not in node_index:
            raise ValueError(f"Outcome '{outcome}' not found in node_names for dataset '{dataset}'")
        outcome_index = node_index[outcome]
        tau_c = runner.dataset_config.get("soft_fairness_settings", {}).get("tau_c", 5.0)

        pi_sigma = compute_edge_proxy_fractions(
            df_structure_discovery.values,
            protected_attr_indices=protected_indices,
            tau_c=tau_c,
            min_group_size=10,
            var_names=node_names,
        )
        rho_adj = compute_outcome_relevance(df_structure_discovery.values, outcome_index)

        rows: List[Dict[str, Any]] = []
        for edge_v1, edge_v2 in skeleton_edges:
            idx_v1 = node_index[edge_v1]
            idx_v2 = node_index[edge_v2]
            pi_val = float(pi_sigma[idx_v1, idx_v2])
            rho_v1 = float(rho_adj.get(idx_v1, 0.0))
            rho_v2 = float(rho_adj.get(idx_v2, 0.0))
            pruning_score = float(pi_val * max(rho_v1, rho_v2))
            rows.append(
                {
                    "dataset": dataset,
                    "edge_v1": edge_v1,
                    "edge_v2": edge_v2,
                    "pi_sigma": pi_val,
                    "rho_adj_v1": rho_v1,
                    "rho_adj_v2": rho_v2,
                    "pruning_score": pruning_score,
                    "cpdag_source": cpdag_source,
                }
            )

    df_scores = pd.DataFrame(rows)
    if df_scores.empty:
        return df_scores
    return df_scores.sort_values(
        ["pruning_score", "edge_v1", "edge_v2"], ascending=[False, True, True]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CPDAG skeleton pruning scores for selected datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to analyze. Use one or more of: law dutch bank compas.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(args.datasets)
    results_root = Path(args.results_root)

    if args.baseline_run_dir and len(datasets) != 1:
        raise ValueError("--baseline-run-dir can only be used with a single dataset")

    all_frames: List[pd.DataFrame] = []
    for dataset in datasets:
        print(f"\n=== Pruning score analysis for {dataset} ===")
        frame = analyze_dataset(
            dataset=dataset,
            results_root=results_root,
            baseline_run_dir=Path(args.baseline_run_dir) if args.baseline_run_dir else None,
        )
        if frame.empty:
            print("No edges found in CPDAG skeleton.")
            continue

        display_cols = [
            "edge_v1",
            "edge_v2",
            "pi_sigma",
            "rho_adj_v1",
            "rho_adj_v2",
            "pruning_score",
        ]
        print(frame[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        all_frames.append(frame)

    out_path = results_root / "pruning_score_analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined[
            [
                "dataset",
                "edge_v1",
                "edge_v2",
                "pi_sigma",
                "rho_adj_v1",
                "rho_adj_v2",
                "pruning_score",
            ]
        ]
        combined.to_csv(out_path, index=False)
        print(f"\nSaved pruning score analysis to: {out_path}")
    else:
        pd.DataFrame(
            columns=[
                "dataset",
                "edge_v1",
                "edge_v2",
                "pi_sigma",
                "rho_adj_v1",
                "rho_adj_v2",
                "pruning_score",
            ]
        ).to_csv(out_path, index=False)
        print(f"\nNo rows produced; wrote empty CSV to: {out_path}")


if __name__ == "__main__":
    main()
