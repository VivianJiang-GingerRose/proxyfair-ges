"""Analyze post-hoc pruning sweep outputs and select paper-ready tau values.

This script produces three outputs from a posthoc pruning sweep CSV:
1) Full tau table with compact columns and paper_role labels.
2) Non-dominated tau table under strict dominance rule.
3) Single main-table tau chosen from the non-dominated set.

Dominance rule (j dominates i):
- |TE|_j < |TE|_i
- |PSE|_j < |PSE|_i
- AUROC_j >= AUROC_i

Main-table selection among non-dominated taus:
- Minimize |PSE|
- Tie-break by maximizing AUROC
- Final tie-break by smallest tau_p
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _mean_and_std(values: Iterable[float]) -> Tuple[float, float]:
    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _parse_cpdag_edge_count_from_log(log_path: Path) -> int | None:
    """Return directed+undirected edge count from the last CPDAG block in a baseline log."""
    if not log_path.exists():
        return None

    directed_re = re.compile(r"^\s*\d+\.\s+(.+?)\s+-->\s+(.+?)\s*$")
    undirected_re = re.compile(r"^\s*\d+\.\s+(.+?)\s+--\s+(.+?)\s*$")

    blocks: list[tuple[int, int]] = []
    directed = 0
    undirected = 0
    in_directed = False
    in_undirected = False

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line == "Directed edges:":
            if directed or undirected:
                blocks.append((directed, undirected))
            directed = 0
            undirected = 0
            in_directed = True
            in_undirected = False
            continue
        if line == "Undirected edges:":
            in_directed = False
            in_undirected = True
            continue
        if line.startswith("DAG edges:"):
            in_directed = False
            in_undirected = False
            continue

        if in_directed and directed_re.match(raw_line):
            directed += 1
        elif in_undirected and undirected_re.match(raw_line):
            undirected += 1

    if directed or undirected:
        blocks.append((directed, undirected))
    if not blocks:
        return None

    d_last, u_last = blocks[-1]
    return int(d_last + u_last)


def _ensure_edge_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # New sweep schema: already contains these diagnostics.
    if "n_edges_before_mean" in out.columns:
        out["n_edges_before"] = out["n_edges_before_mean"].astype(float)
    else:
        inferred: Dict[str, float] = {}
        for _, row in out.iterrows():
            baseline_source = str(row.get("baseline_cpdag_source", "")).strip()
            if baseline_source not in inferred:
                inferred_count = None
                if baseline_source:
                    inferred_count = _parse_cpdag_edge_count_from_log(Path(baseline_source))
                inferred[baseline_source] = float(inferred_count) if inferred_count is not None else float("nan")
            out.loc[row.name, "n_edges_before"] = inferred[baseline_source]

    # Historical sweep files may not include a parseable baseline edge count.
    # Fallback to the max observed post-pruning edge count per baseline source,
    # which is a safe lower-bound proxy for pre-pruning edge count.
    if out["n_edges_before"].isna().any():
        baseline_col = "baseline_cpdag_source" if "baseline_cpdag_source" in out.columns else None
        if baseline_col is not None:
            grouped_max = out.groupby(baseline_col)["n_edges_mean"].transform("max")
            out["n_edges_before"] = out["n_edges_before"].fillna(grouped_max)
        else:
            out["n_edges_before"] = out["n_edges_before"].fillna(out["n_edges_mean"].max())

    out["n_edges_after"] = out["n_edges_mean"].astype(float)

    if "n_pruned_edges_mean" in out.columns:
        out["n_pruned_edges"] = out["n_pruned_edges_mean"].astype(float)
    else:
        out["n_pruned_edges"] = out["n_edges_before"] - out["n_edges_after"]

    return out


def _validate_required(df: pd.DataFrame) -> None:
    required = [
        "tau_p",
        "n_edges_mean",
        "overall_te_abs",
        "overall_pse_abs",
        "overall_nde_abs",
        "overall_dpd",
        "overall_eo",
        "auroc",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sweep columns: {missing}")


def _compute_dominance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_dominated"] = False
    out["dominance_witness_tau"] = np.nan

    for i, row_i in out.iterrows():
        te_i = float(row_i["overall_te_abs"])
        pse_i = float(row_i["overall_pse_abs"])
        auroc_i = float(row_i["auroc"])

        witness = None
        for j, row_j in out.iterrows():
            if i == j:
                continue
            if (
                float(row_j["overall_te_abs"]) < te_i
                and float(row_j["overall_pse_abs"]) < pse_i
                and float(row_j["auroc"]) >= auroc_i
            ):
                witness = float(row_j["tau_p"])
                break

        if witness is not None:
            out.loc[i, "is_dominated"] = True
            out.loc[i, "dominance_witness_tau"] = witness

    return out


def _pick_main_table_tau(frontier: pd.DataFrame) -> float:
    ranked = frontier.sort_values(
        by=["overall_pse_abs", "auroc", "tau_p"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["tau_p"])


def _build_compact_table(df: pd.DataFrame, main_tau: float) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "tau_p": df["tau_p"].astype(float),
            "selected_dag_or_mean": "mean_over_dags",
            "n_edges": df["n_edges_after"].astype(float),
            "n_pruned": df["n_pruned_edges"].astype(float),
            "|TE|": df["overall_te_abs"].astype(float),
            "|PSE|": df["overall_pse_abs"].astype(float),
            "|NDE|": df["overall_nde_abs"].astype(float),
            "DPD": df["overall_dpd"].astype(float),
            "EO": df["overall_eo"].astype(float),
            "AUROC": df["auroc"].astype(float),
            "paper_role": "dominated",
        }
    )

    frontier_mask = ~df["is_dominated"].astype(bool)
    out.loc[frontier_mask, "paper_role"] = "pareto_frontier"
    out.loc[np.isclose(out["tau_p"], main_tau), "paper_role"] = "main_table_selected"
    return out.sort_values(by="tau_p", ascending=True, kind="mergesort").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze posthoc pruning sweep metrics for paper selection.")
    parser.add_argument(
        "--sweep-csv",
        required=True,
        help="Path to posthoc_pruning_sweep_metrics.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to the sweep CSV parent directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sweep_csv = Path(args.sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"Sweep CSV not found: {sweep_csv}")

    out_dir = Path(args.out_dir) if args.out_dir else sweep_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(sweep_csv)
    _validate_required(raw)
    enriched = _ensure_edge_diagnostics(raw)

    critical = ["overall_te_abs", "overall_pse_abs", "auroc"]
    if enriched[critical].isna().any().any():
        missing_rows = enriched[enriched[critical].isna().any(axis=1)][["tau_p"] + critical]
        raise ValueError(
            "Missing values in critical dominance columns:\n"
            + missing_rows.to_string(index=False)
        )

    scored = _compute_dominance(enriched)
    frontier = scored[~scored["is_dominated"].astype(bool)].copy()
    if frontier.empty:
        raise RuntimeError("No non-dominated taus found; cannot select main-table tau.")

    main_tau = _pick_main_table_tau(frontier)
    compact = _build_compact_table(scored, main_tau)

    full_path = out_dir / "posthoc_pruning_compas_tau_full_table.csv"
    frontier_path = out_dir / "posthoc_pruning_compas_tau_pareto_frontier.csv"
    selected_path = out_dir / "posthoc_pruning_compas_tau_main_selected.csv"

    compact.to_csv(full_path, index=False)
    compact[compact["paper_role"].isin(["pareto_frontier", "main_table_selected"])].to_csv(
        frontier_path,
        index=False,
    )
    compact[compact["paper_role"] == "main_table_selected"].to_csv(selected_path, index=False)

    n_total = int(compact.shape[0])
    n_frontier = int(compact[compact["paper_role"].isin(["pareto_frontier", "main_table_selected"])].shape[0])
    n_dominated = int(compact[compact["paper_role"] == "dominated"].shape[0])

    print("Posthoc pruning sweep analysis complete.")
    print(f"Rows total: {n_total}")
    print(f"Rows dominated: {n_dominated}")
    print(f"Rows frontier (including selected): {n_frontier}")
    print(f"Main table tau: {main_tau:.10g}")
    print(f"Wrote: {full_path}")
    print(f"Wrote: {frontier_path}")
    print(f"Wrote: {selected_path}")


if __name__ == "__main__":
    main()
