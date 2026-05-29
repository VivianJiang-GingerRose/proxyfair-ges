from __future__ import annotations

import datetime
import json
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.faircausal.core.counterfactual_fairness_runner import select_fairest_dag
from src.faircausal.core.fairness_scoring import (
    compute_edge_proxy_fractions,
    compute_outcome_relevance,
    compute_symmetric_penalty_from_pi,
)
from src.faircausal.core.ges_runner_causal_learn import run_ges
from src.faircausal.data.synthetic.generator import generate_synthetic_dataset


@dataclass
class FairnessQuantityCache:
    pi_sigma: np.ndarray
    rho_adj: Dict[int, float]
    psi_sym: Dict[Tuple[int, int], float]



def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)



def _get_git_hash() -> str:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return completed.stdout.strip()
    except Exception:
        return "unknown"



def _validate_discrete_input(data: np.ndarray) -> None:
    if data.ndim != 2:
        raise ValueError("Input data must be 2D")
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError("Input data must be integer encoded")
    if np.isnan(data).any():
        raise ValueError("Input data must not contain NaNs")
    unique_counts = [len(np.unique(data[:, j])) for j in range(data.shape[1])]
    if any(c > 100 for c in unique_counts):
        raise ValueError("Input appears non-categorical: a column has >100 unique values")



def _edge_set(edges: Sequence[Tuple[str, str]]) -> set[Tuple[str, str]]:
    return {(str(a), str(b)) for a, b in edges}


def _format_final_dag(edges: Sequence[Tuple[str, str]]) -> str:
    """Serialize learned directed edges into a stable, single-cell string."""
    canonical = sorted(_edge_set(edges))
    return "|".join(f"{src}->{dst}" for src, dst in canonical)


def _format_cpdag(
    directed_edges: Sequence[Tuple[str, str]],
    undirected_edges: Sequence[Tuple[str, str]],
) -> str:
    """Serialize a CPDAG into a stable, single-cell string.

    Directed edges are formatted as ``A->B`` and undirected edges as ``A--B``.
    Within each group the canonical (alphabetically-sorted) order is used;
    directed edges come first, then undirected.
    """
    directed_part = sorted(_edge_set(directed_edges))
    # Normalise undirected pairs so the smaller name always comes first.
    undirected_part = sorted(
        (min(a, b), max(a, b)) for a, b in undirected_edges
    )
    tokens = [f"{src}->{dst}" for src, dst in directed_part] + [
        f"{a}--{b}" for a, b in undirected_part
    ]
    return "|".join(tokens)



def compute_shd(true_edges: Sequence[Tuple[str, str]], learned_edges: Sequence[Tuple[str, str]]) -> int:
    t = _edge_set(true_edges)
    l = _edge_set(learned_edges)

    shd = 0
    for e in t:
        if e in l:
            continue
        if (e[1], e[0]) in l:
            shd += 1
        else:
            shd += 1

    for e in l:
        if e in t:
            continue
        if (e[1], e[0]) in t:
            continue
        shd += 1

    return int(shd)



def _precision_recall(true_subset: set[Tuple[str, str]], learned_subset: set[Tuple[str, str]]) -> Tuple[float, float]:
    tp = len(true_subset & learned_subset)
    precision = tp / len(learned_subset) if learned_subset else 1.0
    recall = tp / len(true_subset) if true_subset else 1.0
    return float(precision), float(recall)



def _subgraph_shd(
    true_edges: Sequence[Tuple[str, str]],
    learned_edges: Sequence[Tuple[str, str]],
    var_subset: set[str],
) -> int:
    """SHD restricted to edges where both endpoints are in var_subset."""
    true_sub = [(s, d) for s, d in true_edges if s in var_subset and d in var_subset]
    learned_sub = [(s, d) for s, d in learned_edges if s in var_subset and d in var_subset]
    return compute_shd(true_sub, learned_sub)



def _split_edges_fair_other(
    edges: Sequence[Tuple[str, str]], fairness_names: set[str]
) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]]]:
    fair = set()
    other = set()
    for src, dst in _edge_set(edges):
        if src in fairness_names or dst in fairness_names:
            fair.add((src, dst))
        else:
            other.add((src, dst))
    return fair, other



def compute_counterfactual_effects(
    learned_edges: Sequence[Tuple[str, str]],
    data: np.ndarray,
    variable_names: Sequence[str],
    protected_index: int,
    proxy_index: int,
    outcome_index: int,
    seed: int,
) -> Dict[str, float]:
    """
    Compute counterfactual TE, NDE, and NIE using hard interventions and empirical CPTs.

      TE  = E[Y | do(S=1)] - E[Y | do(S=0)]
      NDE = E[Y | do(S=1), V=V(do(S=0))] - E[Y | do(S=0)]
            (S intervened to 1, but V held at its natural value under S=0)
      NIE = TE - NDE   (proxy-mediated portion, the causal fairness PSE through V)

    Mirrors the counterfactual total effect in counterfactual_fairness_runner
    (y_total_cf - y_factual). Returns nan dict if the learned graph is cyclic.
    """
    nan_result: Dict[str, float] = {"cf_te": float("nan"), "cf_nde": float("nan"), "cf_nie": float("nan")}

    graph = nx.DiGraph()
    graph.add_nodes_from(variable_names)
    graph.add_edges_from(learned_edges)
    if not nx.is_directed_acyclic_graph(graph):
        return nan_result

    order = list(nx.topological_sort(graph))
    name_to_idx = {name: i for i, name in enumerate(variable_names)}
    protected_name = variable_names[protected_index]
    proxy_name = variable_names[proxy_index]
    n_samples = data.shape[0]

    # Build empirical CPTs from observed data
    marginal: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    cpt: Dict[str, Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]]] = {}
    for name in variable_names:
        node_idx = name_to_idx[name]
        node_vals = data[:, node_idx].astype(int)
        vals, counts = np.unique(node_vals, return_counts=True)
        marginal[name] = (vals.astype(int), counts.astype(np.float64) / float(counts.sum()))
        parent_names = list(graph.predecessors(name))
        if not parent_names:
            continue
        parent_idxs = [name_to_idx[p] for p in parent_names]
        buckets: Dict[Tuple[int, ...], List[int]] = {}
        for row_idx in range(n_samples):
            key = tuple(int(data[row_idx, p]) for p in parent_idxs)
            buckets.setdefault(key, []).append(int(node_vals[row_idx]))
        node_cpt: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]] = {}
        for key, bvals in buckets.items():
            v, c = np.unique(np.asarray(bvals, dtype=int), return_counts=True)
            node_cpt[key] = (v.astype(int), c.astype(np.float64) / float(c.sum()))
        cpt[name] = node_cpt

    rng = np.random.default_rng(seed)

    def _forward_sample(do_s: int, fix_proxy: Optional[np.ndarray] = None) -> np.ndarray:
        sim = np.zeros((n_samples, len(variable_names)), dtype=int)
        for row_idx in range(n_samples):
            for name in order:
                node_idx = name_to_idx[name]
                if name == protected_name:
                    sim[row_idx, node_idx] = int(do_s)
                    continue
                if fix_proxy is not None and name == proxy_name:
                    sim[row_idx, node_idx] = int(fix_proxy[row_idx])
                    continue
                parent_names = list(graph.predecessors(name))
                if not parent_names:
                    v, p = marginal[name]
                else:
                    key = tuple(int(sim[row_idx, name_to_idx[pp]]) for pp in parent_names)
                    vp = cpt.get(name, {}).get(key)
                    v, p = vp if vp is not None else marginal[name]
                sim[row_idx, node_idx] = int(rng.choice(v, p=p))
        return sim

    # E[Y | do(S=0)]: factual baseline + natural V values under S=0
    sim_s0 = _forward_sample(do_s=0)
    y_s0 = float(np.mean(sim_s0[:, outcome_index]))
    v_natural_s0 = sim_s0[:, proxy_index]

    # E[Y | do(S=1)]: total intervention
    sim_s1 = _forward_sample(do_s=1)
    y_s1 = float(np.mean(sim_s1[:, outcome_index]))

    # E[Y | do(S=1), V=V(do(S=0))]: NDE — S intervened but V blocked at natural S=0 value
    sim_nde = _forward_sample(do_s=1, fix_proxy=v_natural_s0)
    y_nde_val = float(np.mean(sim_nde[:, outcome_index]))

    te = y_s1 - y_s0
    nde = y_nde_val - y_s0
    nie = te - nde

    return {"cf_te": te, "cf_nde": nde, "cf_nie": nie}


def estimate_pse_from_edges(
    learned_edges: Sequence[Tuple[str, str]],
    data: np.ndarray,
    variable_names: Sequence[str],
    protected_name: str,
    proxy_name: str,
    outcome_name: str,
) -> Dict[str, float]:
    g = nx.DiGraph()
    g.add_nodes_from(variable_names)
    g.add_edges_from(learned_edges)

    idx = {name: i for i, name in enumerate(variable_names)}
    parents: Dict[str, List[str]] = {n: [] for n in variable_names}
    for src, dst in learned_edges:
        parents[dst].append(src)

    edge_coef: Dict[Tuple[str, str], float] = {}

    for node in variable_names:
        pa = parents[node]
        if not pa:
            continue
        y = data[:, idx[node]].astype(np.float64)
        X = data[:, [idx[p] for p in pa]].astype(np.float64)
        Xd = np.column_stack([np.ones(X.shape[0]), X])
        coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
        for p, c in zip(pa, coef[1:]):
            edge_coef[(p, node)] = float(c)

    total = 0.0
    if protected_name in g.nodes and outcome_name in g.nodes:
        for path in nx.all_simple_paths(g, source=protected_name, target=outcome_name, cutoff=len(variable_names)):
            prod = 1.0
            valid = True
            for a, b in zip(path[:-1], path[1:]):
                if (a, b) not in edge_coef:
                    valid = False
                    break
                prod *= edge_coef[(a, b)]
            if valid:
                total += prod

    direct = float(edge_coef.get((protected_name, outcome_name), 0.0))

    indirect_proxy = 0.0
    if protected_name in g.nodes and outcome_name in g.nodes and proxy_name in g.nodes:
        for path in nx.all_simple_paths(g, source=protected_name, target=outcome_name, cutoff=len(variable_names)):
            if proxy_name not in path:
                continue
            if len(path) < 3:
                continue
            prod = 1.0
            valid = True
            for a, b in zip(path[:-1], path[1:]):
                if (a, b) not in edge_coef:
                    valid = False
                    break
                prod *= edge_coef[(a, b)]
            if valid:
                indirect_proxy += prod

    return {
        "total": float(total),
        "direct": float(direct),
        "indirect_proxy": float(indirect_proxy),
    }


def _estimate_edge_coefficients(
    learned_edges: Sequence[Tuple[str, str]],
    data: np.ndarray,
    variable_names: Sequence[str],
) -> Dict[Tuple[str, str], float]:
    idx = {name: i for i, name in enumerate(variable_names)}
    parents: Dict[str, List[str]] = {name: [] for name in variable_names}
    for src, dst in learned_edges:
        if src in idx and dst in idx:
            parents[dst].append(src)

    edge_coef: Dict[Tuple[str, str], float] = {}
    for node, pa in parents.items():
        if not pa:
            continue
        y = data[:, idx[node]].astype(np.float64)
        X = data[:, [idx[p] for p in pa]].astype(np.float64)
        Xd = np.column_stack([np.ones(X.shape[0]), X])
        coef = np.linalg.lstsq(Xd, y, rcond=None)[0]
        for p, c in zip(pa, coef[1:]):
            edge_coef[(p, node)] = float(c)
    return edge_coef


def estimate_path_pse(
    learned_edges: Sequence[Tuple[str, str]],
    data: np.ndarray,
    variable_names: Sequence[str],
    path: Tuple[str, str, str],
) -> float:
    learned_set = _edge_set(learned_edges)
    src, proxy, outcome = path
    if (src, proxy) not in learned_set or (proxy, outcome) not in learned_set:
        return 0.0
    edge_coef = _estimate_edge_coefficients(learned_edges, data, variable_names)
    return float(edge_coef.get((src, proxy), 0.0) * edge_coef.get((proxy, outcome), 0.0))



def _compute_downstream_metrics(
    data: np.ndarray,
    variable_names: Sequence[str],
    learned_edges: Sequence[Tuple[str, str]],
    outcome_index: int,
    protected_index: int,
    seed: int,
) -> Dict[str, float]:
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = max(1, int(0.7 * n))
    tr = perm[:split]
    te = perm[split:]

    y_name = variable_names[outcome_index]
    parents_y = [src for src, dst in learned_edges if dst == y_name]

    y_train = data[tr, outcome_index].astype(np.float64)
    y_test = data[te, outcome_index].astype(np.float64)

    if not parents_y:
        pred = np.full_like(y_test, y_train.mean() if y_train.size else 0.0, dtype=np.float64)
        mse = float(np.mean((pred - y_test) ** 2)) if y_test.size else 0.0
        return {"mse": mse, "dp_gap": 0.0, "cf_violation": 0.0}

    idx = {n: i for i, n in enumerate(variable_names)}
    x_cols = [idx[nm] for nm in parents_y]

    X_train = data[tr][:, x_cols].astype(np.float64)
    X_test = data[te][:, x_cols].astype(np.float64)

    coef = np.linalg.lstsq(np.column_stack([np.ones(X_train.shape[0]), X_train]), y_train, rcond=None)[0]
    pred = np.column_stack([np.ones(X_test.shape[0]), X_test]) @ coef

    mse = float(np.mean((pred - y_test) ** 2)) if y_test.size else 0.0

    s_test = data[te, protected_index] if te.size else np.array([], dtype=np.int64)
    if s_test.size and np.any(s_test == 1) and np.any(s_test == 0):
        dp_gap = float(abs(pred[s_test == 1].mean() - pred[s_test == 0].mean()))
    else:
        dp_gap = 0.0

    cf_violation = 0.0
    protected_name = variable_names[protected_index]
    if protected_name in parents_y and X_test.size:
        s_col = parents_y.index(protected_name)
        X_cf = X_test.copy()
        X_cf[:, s_col] = 1.0 - X_cf[:, s_col]
        pred_cf = np.column_stack([np.ones(X_cf.shape[0]), X_cf]) @ coef
        cf_violation = float(np.mean(np.abs(pred - pred_cf)))

    return {
        "mse": mse,
        "dp_gap": float(dp_gap),
        "cf_violation": float(cf_violation),
    }



def build_fairness_cache(
    data: np.ndarray,
    variable_names: List[str],
    protected_indices: List[int],
    outcome_index: int,
    tau_c: float,
) -> FairnessQuantityCache:
    pi_sigma = compute_edge_proxy_fractions(
        data,
        protected_attr_indices=protected_indices,
        tau_c=tau_c,
        min_group_size=10,
        var_names=variable_names,
    )
    rho_adj = compute_outcome_relevance(data, outcome_index)
    psi_sym = compute_symmetric_penalty_from_pi(pi_sigma, rho_adj)
    return FairnessQuantityCache(
        pi_sigma=pi_sigma,
        rho_adj=rho_adj,
        psi_sym=psi_sym,
    )



def _algorithm_to_params(
    algorithm: str,
    protected_indices: List[int],
    outcome_index: int,
    proxy_index: int,
    n_vars: int,
    lambda_f: float,
    enable_temporal_blacklist: bool,
    variable_names: Optional[List[str]] = None,
    proxy_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    def _displacement_blacklist() -> np.ndarray:
        if variable_names is None:
            raise ValueError("variable_names are required for displacement blacklist")
        name_to_idx = {name: i for i, name in enumerate(variable_names)}
        allowed_edges = {
            ("C", "S1"),
            ("C", "S2"),
            ("S1", "V1"),
            ("S2", "V2"),
            ("V1", "Y"),
            ("V2", "Y"),
            ("Z", "Y"),
        }
        blacklist = np.ones((n_vars, n_vars), dtype=bool)
        for i in range(n_vars):
            blacklist[i, i] = False
        for src, dst in allowed_edges:
            if src in name_to_idx and dst in name_to_idx:
                blacklist[name_to_idx[src], name_to_idx[dst]] = False
        return blacklist

    def _temporal_blacklist() -> np.ndarray:
        blacklist = np.zeros((n_vars, n_vars), dtype=bool)

        # Y is terminal: outcome cannot cause any variable.
        blacklist[outcome_index, :] = True
        blacklist[outcome_index, outcome_index] = False

        # Protected attributes are predetermined/exogenous in this setup.
        for s in protected_indices:
            blacklist[:, s] = True
            blacklist[s, s] = False

        # Proxy is downstream of non-outcome variables: only proxy -> outcome remains allowed.
        blacklist[proxy_index, :] = True
        blacklist[proxy_index, outcome_index] = False
        blacklist[proxy_index, proxy_index] = False
        return blacklist

    is_displacement = variable_names is not None and {"C", "S1", "S2", "V1", "V2", "Z", "Y"}.issubset(variable_names)

    if algorithm == "vanilla_ges":
        blacklist = (
            _displacement_blacklist()
            if is_displacement and enable_temporal_blacklist
            else _temporal_blacklist() if enable_temporal_blacklist
            else None
        )
        return {
            "experiment_type": "temporal_constraints" if blacklist is not None else "baseline",
            "blacklist_matrix": blacklist,
            "fairness_lambda": None,
            "fairness_tau_c": 5.0,
            "protected_attr_indices": None,
            "outcome_index": None,
        }

    if algorithm == "hard_constraints":
        blacklist = (
            _displacement_blacklist()
            if is_displacement and enable_temporal_blacklist
            else _temporal_blacklist() if enable_temporal_blacklist
            else np.zeros((n_vars, n_vars), dtype=bool)
        )

        # --- Fairness constraint ---
        for s in protected_indices:
            blacklist[s, outcome_index] = True

        return {
            "experiment_type": "domain_knowledge",
            "blacklist_matrix": blacklist,
            "fairness_lambda": None,
            "fairness_tau_c": 5.0,
            "protected_attr_indices": None,
            "outcome_index": None,
        }

    if algorithm in {"fair_mec_phase1", "fair_mec_full", "fair_mec_marginal_only"}:
        blacklist = (
            _displacement_blacklist()
            if is_displacement and enable_temporal_blacklist
            else _temporal_blacklist() if enable_temporal_blacklist
            else None
        )
        return {
            "experiment_type": "soft_fairness",
            "blacklist_matrix": blacklist,
            "fairness_lambda": float(lambda_f),
            "fairness_tau_c": 5.0,
            "protected_attr_indices": protected_indices,
            "outcome_index": outcome_index,
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")



def _extract_learned_edges(
    algorithm: str,
    ges_result: Dict[str, Any],
    data: np.ndarray,
    variable_names: List[str],
    protected_indices: List[int],
    outcome_index: int,
    lambda_f: float,
    beta: float,
) -> List[Tuple[str, str]]:
    if algorithm != "fair_mec_full":
        return list(ges_result.get("dowhy_edges", []))

    stage2 = select_fairest_dag(
        cpdag=ges_result["G"],
        data=data,
        node_names=variable_names,
        protected_attr_indices=protected_indices,
        outcome_index=outcome_index,
        fairness_lambda=lambda_f,
        beta=beta,
    )
    dag_edges = stage2.get("dag_edges", [])
    if not dag_edges:
        return list(ges_result.get("dowhy_edges", []))
    return list(dag_edges)



def _compute_ml_fairness_metrics(
    data: np.ndarray,
    variable_names: Sequence[str],
    learned_edges: Sequence[Tuple[str, str]],
    outcome_index: int,
    protected_index: int,
    seed: int,
) -> Dict[str, float]:
    """Fit a logistic regression on parents of Y (from the learned DAG) and compute
    downstream ML performance metrics and group-fairness statistics.

    Y is binarised via a median split so results are robust to the number of bins.
    Group-fairness stats (TPR/FPR gaps, etc.) are always stratified by the *observed*
    protected attribute, regardless of whether S appears in the learned DAG.
    """
    nan_result: Dict[str, float] = {
        "auc_roc": float("nan"),
        "auc_roc_std": float("nan"),
        "auc_roc_var": float("nan"),
        "f1": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "dp_gap_lr": float("nan"),
        "tpr_gap": float("nan"),
        "fpr_gap": float("nan"),
        "disparate_impact": float("nan"),
        "ppv_gap": float("nan"),
    }

    def _simulate_from_learned_dag() -> Optional[np.ndarray]:
        # Build a DAG over all variables and verify acyclicity before sampling.
        graph = nx.DiGraph()
        graph.add_nodes_from(variable_names)
        graph.add_edges_from(learned_edges)
        if not nx.is_directed_acyclic_graph(graph):
            return None

        order = list(nx.topological_sort(graph))
        name_to_idx = {name: i for i, name in enumerate(variable_names)}
        n_samples = int(data.shape[0])

        # Empirical discrete support for each variable from observed data.
        value_support: Dict[str, np.ndarray] = {
            name: np.unique(data[:, name_to_idx[name]]).astype(int)
            for name in variable_names
        }

        # Estimate empirical CPTs P(node | parents) from observed data.
        cpt: Dict[str, Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]]] = {}
        marginal: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for name in variable_names:
            node_idx = name_to_idx[name]
            node_vals = data[:, node_idx].astype(int)
            vals, counts = np.unique(node_vals, return_counts=True)
            probs = counts.astype(np.float64) / float(np.sum(counts))
            marginal[name] = (vals.astype(int), probs)

            parent_names = list(graph.predecessors(name))
            if not parent_names:
                continue

            parent_idxs = [name_to_idx[p] for p in parent_names]
            buckets: Dict[Tuple[int, ...], List[int]] = {}
            for row_idx in range(data.shape[0]):
                key = tuple(int(v) for v in data[row_idx, parent_idxs])
                buckets.setdefault(key, []).append(int(node_vals[row_idx]))

            node_cpt: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]] = {}
            for key, bucket_vals in buckets.items():
                vals_k, counts_k = np.unique(np.asarray(bucket_vals, dtype=int), return_counts=True)
                probs_k = counts_k.astype(np.float64) / float(np.sum(counts_k))
                node_cpt[key] = (vals_k.astype(int), probs_k)
            cpt[name] = node_cpt

        rng = np.random.default_rng(seed)
        sim = np.zeros((n_samples, len(variable_names)), dtype=int)

        for row_idx in range(n_samples):
            for name in order:
                node_idx = name_to_idx[name]
                parent_names = list(graph.predecessors(name))
                if not parent_names:
                    vals, probs = marginal[name]
                else:
                    key = tuple(int(sim[row_idx, name_to_idx[p]]) for p in parent_names)
                    vals_probs = cpt.get(name, {}).get(key)
                    if vals_probs is None:
                        # Backoff to empirical marginal when parent config is unseen.
                        vals, probs = marginal[name]
                    else:
                        vals, probs = vals_probs

                sim[row_idx, node_idx] = int(rng.choice(vals, p=probs))

        return sim

    simulated_data = _simulate_from_learned_dag()
    if simulated_data is None:
        return nan_result

    # ---- binarise outcome via median split ----
    y_all = simulated_data[:, outcome_index].astype(np.float64)
    median_val = float(np.median(y_all))
    y_bin = (y_all >= median_val).astype(int)

    # ---- feature matrix: parents of Y in learned DAG ----
    y_name = variable_names[outcome_index]
    parents_y = [src for src, dst in learned_edges if dst == y_name]

    if not parents_y:
        return nan_result

    idx = {nm: i for i, nm in enumerate(variable_names)}
    x_cols = [idx[nm] for nm in parents_y]

    X_all = simulated_data[:, x_cols].astype(np.float64)
    if X_all.shape[0] == 0 or len(np.unique(y_bin)) < 2:
        return nan_result

    def _safe_mean(values: List[float]) -> float:
        finite = [float(v) for v in values if np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.mean(finite))

    def _safe_std(values: List[float]) -> float:
        finite = [float(v) for v in values if np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.std(finite))

    def _safe_var(values: List[float]) -> float:
        finite = [float(v) for v in values if np.isfinite(v)]
        if not finite:
            return float("nan")
        return float(np.var(finite))

    auc_scores: List[float] = []
    f1_scores: List[float] = []
    accuracy_scores: List[float] = []
    balanced_scores: List[float] = []
    dp_gap_scores: List[float] = []
    tpr_gap_scores: List[float] = []
    fpr_gap_scores: List[float] = []
    di_scores: List[float] = []
    ppv_gap_scores: List[float] = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X_all, y_bin):
        X_train = X_all[tr]
        X_test = X_all[te]
        y_train_bin = y_bin[tr]
        y_test_bin = y_bin[te]

        # Guard: each fold should contain both classes, but we fail-safe anyway.
        if len(np.unique(y_train_bin)) < 2 or X_test.shape[0] == 0:
            continue

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
        try:
            clf.fit(X_train, y_train_bin)
        except Exception:
            continue

        y_pred = clf.predict(X_test)
        try:
            prob_pos = clf.predict_proba(X_test)[:, 1]
        except Exception:
            prob_pos = y_pred.astype(np.float64)

        if len(np.unique(y_test_bin)) < 2:
            auc_scores.append(float("nan"))
        else:
            auc_scores.append(float(roc_auc_score(y_test_bin, prob_pos)))

        accuracy_scores.append(float(accuracy_score(y_test_bin, y_pred)))
        balanced_scores.append(float(balanced_accuracy_score(y_test_bin, y_pred)))
        f1_scores.append(float(f1_score(y_test_bin, y_pred, zero_division=0)))

        # ---- group-fairness stats on each fold, then average across folds ----
        s_test = simulated_data[te, protected_index]
        mask1 = s_test == 1
        mask0 = s_test == 0
        pred_pos = y_pred == 1

        def _safe_rate(pred_mask: np.ndarray, group_mask: np.ndarray) -> float:
            if not group_mask.any():
                return float("nan")
            return float(pred_mask[group_mask].mean())

        rate1 = _safe_rate(pred_pos, mask1)
        rate0 = _safe_rate(pred_pos, mask0)
        dp_gap_scores.append(float(abs(rate1 - rate0)) if not (np.isnan(rate1) or np.isnan(rate0)) else float("nan"))
        di_scores.append(float(rate1 / rate0) if (not np.isnan(rate0) and rate0 > 0) else float("nan"))

        def _conditional_rate(pred_mask: np.ndarray, group_mask: np.ndarray, label_mask: np.ndarray) -> float:
            denom_mask = group_mask & label_mask
            if not denom_mask.any():
                return float("nan")
            return float(pred_mask[denom_mask].mean())

        truly_pos = y_test_bin == 1
        truly_neg = y_test_bin == 0

        tpr1 = _conditional_rate(pred_pos, mask1, truly_pos)
        tpr0 = _conditional_rate(pred_pos, mask0, truly_pos)
        tpr_gap_scores.append(float(abs(tpr1 - tpr0)) if not (np.isnan(tpr1) or np.isnan(tpr0)) else float("nan"))

        fpr1 = _conditional_rate(pred_pos, mask1, truly_neg)
        fpr0 = _conditional_rate(pred_pos, mask0, truly_neg)
        fpr_gap_scores.append(float(abs(fpr1 - fpr0)) if not (np.isnan(fpr1) or np.isnan(fpr0)) else float("nan"))

        def _ppv(group_mask: np.ndarray) -> float:
            denom_mask = pred_pos & group_mask
            if not denom_mask.any():
                return float("nan")
            return float((y_test_bin[denom_mask] == 1).mean())

        ppv1 = _ppv(mask1)
        ppv0 = _ppv(mask0)
        ppv_gap_scores.append(float(abs(ppv1 - ppv0)) if not (np.isnan(ppv1) or np.isnan(ppv0)) else float("nan"))

    if not accuracy_scores:
        return nan_result

    return {
        "auc_roc": _safe_mean(auc_scores),
        "auc_roc_std": _safe_std(auc_scores),
        "auc_roc_var": _safe_var(auc_scores),
        "f1": _safe_mean(f1_scores),
        "accuracy": _safe_mean(accuracy_scores),
        "balanced_accuracy": _safe_mean(balanced_scores),
        "dp_gap_lr": _safe_mean(dp_gap_scores),
        "tpr_gap": _safe_mean(tpr_gap_scores),
        "fpr_gap": _safe_mean(fpr_gap_scores),
        "disparate_impact": _safe_mean(di_scores),
        "ppv_gap": _safe_mean(ppv_gap_scores),
    }


def collect_metrics(
    learned_edges: List[Tuple[str, str]],
    syn: Dict[str, Any],
    fairness_cache: FairnessQuantityCache,
    algorithm: str,
    lambda_f: float,
    seed: int,
    phi_target: float,
    N: int,
    n_bins: int,
    possible_parents_y: Optional[List[str]] = None,
    graph_type: str = "dag",
    undirected_edges: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    variable_names = syn["variable_names"]
    protected_indices = syn["protected_indices"]
    outcome_index = syn["outcome_index"]
    proxy_index = syn["proxy_index"]
    legit_index = syn["legit_index"]
    data = syn["data"]

    y_name = variable_names[outcome_index]
    proxy_name = variable_names[proxy_index]
    legit_name = variable_names[legit_index]
    parents_y_lower = [src for src, dst in learned_edges if dst == y_name]
    if possible_parents_y is not None:
        parents_y_upper = [p for p in possible_parents_y if p in variable_names]
    else:
        parents_y_upper = parents_y_lower

    name_to_idx = {n: i for i, n in enumerate(variable_names)}
    proxy_load_pa_y_lower = 0.0
    proxy_load_pa_y_upper = 0.0
    for p in parents_y_lower:
        proxy_load_pa_y_lower += float(fairness_cache.pi_sigma[name_to_idx[p], outcome_index])
    for p in parents_y_upper:
        proxy_load_pa_y_upper += float(fairness_cache.pi_sigma[name_to_idx[p], outcome_index])

    true_edges = syn["true_edges"]
    shd = compute_shd(true_edges, learned_edges)

    fairness_names = {variable_names[i] for i in protected_indices}
    fairness_names.add(y_name)
    fairness_names.add(proxy_name)

    true_fair, true_other = _split_edges_fair_other(true_edges, fairness_names)
    learned_fair, learned_other = _split_edges_fair_other(learned_edges, fairness_names)

    precision_fair, recall_fair = _precision_recall(true_fair, learned_fair)
    precision_other, recall_other = _precision_recall(true_other, learned_other)

    protected_name = variable_names[protected_indices[0]]

    est_pse = estimate_pse_from_edges(
        learned_edges=learned_edges,
        data=data,
        variable_names=variable_names,
        protected_name=protected_name,
        proxy_name=proxy_name,
        outcome_name=y_name,
    )

    true_pse = syn["true_pse"]

    cf_effects = compute_counterfactual_effects(
        learned_edges=learned_edges,
        data=data,
        variable_names=variable_names,
        protected_index=protected_indices[0],
        proxy_index=proxy_index,
        outcome_index=outcome_index,
        seed=seed,
    )

    downstream = _compute_downstream_metrics(
        data=data,
        variable_names=variable_names,
        learned_edges=learned_edges,
        outcome_index=outcome_index,
        protected_index=protected_indices[0],
        seed=seed,
    )

    ml_metrics = _compute_ml_fairness_metrics(
        data=data,
        variable_names=variable_names,
        learned_edges=learned_edges,
        outcome_index=outcome_index,
        protected_index=protected_indices[0],
        seed=seed,
    )

    # --- Selectivity / subgraph metrics ---
    learned_set = _edge_set(learned_edges)
    suppressed_proxy = float((proxy_name, y_name) not in learned_set)
    retained_legit = float((legit_name, y_name) in learned_set)

    undirected_pairs = {
        frozenset((src, dst)) for src, dst in (undirected_edges or []) if src != dst
    }
    proxy_y_undirected = frozenset((proxy_name, y_name)) in undirected_pairs
    legit_y_undirected = frozenset((legit_name, y_name)) in undirected_pairs

    # Lower/upper bounds for CPDAG outputs.
    # For DAG outputs, these collapse to point estimates.
    suppressed_proxy_lower = float(
        (proxy_name, y_name) not in learned_set and not proxy_y_undirected
    )
    suppressed_proxy_upper = float((proxy_name, y_name) not in learned_set)

    retained_legit_lower = float((legit_name, y_name) in learned_set)
    retained_legit_upper = float(
        (legit_name, y_name) in learned_set or legit_y_undirected
    )

    fair_critical_vars = {variable_names[i] for i in protected_indices} | {proxy_name, y_name}
    legit_vars = {legit_name, y_name}
    shd_fair_critical = float(_subgraph_shd(true_edges, learned_edges, fair_critical_vars))
    shd_legit = float(_subgraph_shd(true_edges, learned_edges, legit_vars))

    delta_star_vy = float(fairness_cache.pi_sigma[proxy_index, outcome_index])
    delta_star_zy = float(fairness_cache.pi_sigma[legit_index, outcome_index])
    phi_star_v = float(np.max(fairness_cache.pi_sigma[proxy_index, :]))
    final_dag = _format_final_dag(learned_edges)
    undirected_for_cpdag = list(undirected_edges or [])
    if graph_type == "dag":
        # DAG rows should never serialize undirected tokens.
        undirected_for_cpdag = []
    final_cpdag = _format_cpdag(learned_edges, undirected_for_cpdag)

    return {
        "archetype": int(syn["archetype_id"]),
        "N": int(N),
        "n_bins": int(n_bins),
        "phi_target": float(phi_target),
        "phi_star": float(phi_target),
        "seed": int(seed),
        "algorithm": algorithm,
        "algorithm_family": "ges",
        "graph_type": graph_type,
        "lambda_param": float(lambda_f),
        "shd": float(shd),
        "precision_fair": float(precision_fair),
        "recall_fair": float(recall_fair),
        "precision_other": float(precision_other),
        "recall_other": float(recall_other),
        "proxy_load_pa_y_lower": float(proxy_load_pa_y_lower),
        "proxy_load_pa_y_upper": float(proxy_load_pa_y_upper),
        "suppressed_proxy": suppressed_proxy,
        "retained_legit": retained_legit,
        "suppressed_proxy_lower": suppressed_proxy_lower,
        "suppressed_proxy_upper": suppressed_proxy_upper,
        "retained_legit_lower": retained_legit_lower,
        "retained_legit_upper": retained_legit_upper,
        "shd_fair_critical": shd_fair_critical,
        "shd_legit": shd_legit,
        "delta_star_vy": delta_star_vy,
        "delta_star_zy": delta_star_zy,
        "phi_star_v": phi_star_v,
        "final_dag": final_dag,
        "final_cpdag": final_cpdag,
        "pse_total_error": float(abs(est_pse["total"] - true_pse["total"])),
        "pse_direct_error": float(abs(est_pse["direct"] - true_pse["direct"])),
        "pse_indirect_error": float(abs(est_pse["indirect_proxy"] - true_pse["indirect_proxy"])),
        # Counterfactual causal effects (CPT forward-sampling, mirrors main_runner)
        "cf_te": float(cf_effects["cf_te"]),
        "cf_nde": float(cf_effects["cf_nde"]),
        "cf_nie": float(cf_effects["cf_nie"]),
        # OLS path-coefficient estimates (linear approximation on learned DAG)
        "est_te": float(est_pse["total"]),
        "est_de": float(est_pse["direct"]),
        "est_ie_proxy": float(est_pse["indirect_proxy"]),
        # NDE = TE - PSE_proxy (non-proxy portion); NIE = proxy-mediated portion
        "nde": float(est_pse["total"] - est_pse["indirect_proxy"]),
        "nie": float(est_pse["indirect_proxy"]),
        # Ground-truth causal effects (from the true DGP)
        "true_te": float(true_pse["total"]),
        "true_de": float(true_pse["direct"]),
        "true_ie_proxy": float(true_pse["indirect_proxy"]),
        # Linear-model downstream metrics (existing)
        "dp_gap": float(downstream["dp_gap"]),
        "cf_violation": float(downstream["cf_violation"]),
        "mse": float(downstream["mse"]),
        # Logistic-regression ML performance
        "auc_roc": float(ml_metrics["auc_roc"]),
        "auc_roc_std": float(ml_metrics["auc_roc_std"]),
        "auc_roc_var": float(ml_metrics["auc_roc_var"]),
        "f1": float(ml_metrics["f1"]),
        "accuracy": float(ml_metrics["accuracy"]),
        "balanced_accuracy": float(ml_metrics["balanced_accuracy"]),
        # Group-fairness statistics
        "dp_gap_lr": float(ml_metrics["dp_gap_lr"]),
        "tpr_gap": float(ml_metrics["tpr_gap"]),
        "fpr_gap": float(ml_metrics["fpr_gap"]),
        "disparate_impact": float(ml_metrics["disparate_impact"]),
        "ppv_gap": float(ml_metrics["ppv_gap"]),
        "calibration_phi_achieved": float(syn["calibration"].get("phi_achieved", np.nan)),
        "calibration_phi_r2_achieved": float(syn["calibration"].get("phi_r2_achieved", np.nan)),
        "calibration_phi_nmi_achieved": float(syn["calibration"].get("phi_nmi_achieved", np.nan)),
        "calibration_phi_star_cramers_v_achieved": float(
            syn["calibration"].get("phi_star_cramers_v_achieved", np.nan)
        ),
        "calibration_phi_star_cramers_v_abs_error": float(
            syn["calibration"].get("phi_star_cramers_v_abs_error", np.nan)
        ),
        "min_category_freq": float(
            min(v["min_category_freq"] for v in syn["diagnostics"]["category"].values())
        ),
    }


def collect_displacement_metrics(
    learned_edges: List[Tuple[str, str]],
    syn: Dict[str, Any],
    algorithm: str,
    lambda_f: float,
    seed: int,
    N: int,
    n_bins: int,
    condition: Optional[str] = None,
    penalty_protected_indices: Optional[List[int]] = None,
    graph_type: str = "dag",
    undirected_edges: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    variable_names = syn["variable_names"]
    data = syn["data"]
    true_edges = syn["true_edges"]
    calibration = syn.get("calibration", {})
    paths = syn.get("displacement_paths", [("S1", "V1", "Y"), ("S2", "V2", "Y")])

    pse_s1 = estimate_path_pse(learned_edges, data, variable_names, tuple(paths[0]))
    pse_s2 = estimate_path_pse(learned_edges, data, variable_names, tuple(paths[1]))
    pse_total = float(pse_s1 + pse_s2)
    learned_set = _edge_set(learned_edges)

    undirected_for_cpdag = list(undirected_edges or [])
    if graph_type == "dag":
        undirected_for_cpdag = []

    penalty_names: List[str] = []
    if penalty_protected_indices:
        penalty_names = [variable_names[i] for i in penalty_protected_indices]

    row: Dict[str, Any] = {
        "archetype": int(syn["archetype_id"]),
        "N": int(N),
        "n_bins": int(n_bins),
        "phi_target": float(calibration.get("phi_target", 0.5)),
        "phi_star": float(calibration.get("phi_star_target", 0.5)),
        "seed": int(seed),
        "condition": condition or algorithm,
        "algorithm": algorithm,
        "algorithm_family": "ges",
        "graph_type": graph_type,
        "lambda_param": float(lambda_f),
        "penalty_protected": "|".join(penalty_names),
        "shd": float(compute_shd(true_edges, learned_edges)),
        "pse_s1": float(pse_s1),
        "pse_s2": float(pse_s2),
        "pse_total": pse_total,
        "abs_pse_s1": float(abs(pse_s1)),
        "abs_pse_s2": float(abs(pse_s2)),
        "abs_pse_total": float(abs(pse_total)),
        "retained_z_y": float(("Z", "Y") in learned_set),
        "retained_v1_y": float(("V1", "Y") in learned_set),
        "retained_v2_y": float(("V2", "Y") in learned_set),
        "retained_s1_v1": float(("S1", "V1") in learned_set),
        "retained_s2_v2": float(("S2", "V2") in learned_set),
        "final_dag": _format_final_dag(learned_edges),
        "final_cpdag": _format_cpdag(learned_edges, undirected_for_cpdag),
        "true_pse_s1": float(syn.get("true_pse", {}).get("pse_s1", np.nan)),
        "true_pse_s2": float(syn.get("true_pse", {}).get("pse_s2", np.nan)),
        "true_pse_total": float(syn.get("true_pse", {}).get("pse_total", np.nan)),
        "min_category_freq": float(
            min(v["min_category_freq"] for v in syn["diagnostics"]["category"].values())
        ),
    }
    for key, value in calibration.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            row[f"calibration_{key}"] = float(value)
        else:
            row[f"calibration_{key}"] = value
    return row


def _cpdag_neighbors_of_outcome(
    graph: Any,
    variable_names: List[str],
    outcome_index: int,
) -> List[str]:
    # In CPDAG, any node adjacent to Y can be a possible parent of Y.
    neighbors: List[str] = []
    y_idx = int(outcome_index)
    for i, name in enumerate(variable_names):
        if i == y_idx:
            continue
        if graph.graph[i, y_idx] != 0 or graph.graph[y_idx, i] != 0:
            neighbors.append(name)
    return neighbors


def _is_cpdag_graph(graph: Any, n_vars: int) -> bool:
    # In causal-learn CPDAG representations, an undirected edge appears as
    # non-zero entries in both directions for a node pair.
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if graph.graph[i, j] != 0 and graph.graph[j, i] != 0:
                return True
    return False


def _select_discrete_score_func(experiment_type: str) -> str:
    """Select the scoring function for discrete synthetic experiments.

    All synthetic modes use BDeu for integer-encoded categorical data.
    """
    _ = experiment_type
    return "local_score_BDeu"



def run_single_experiment(
    archetype_id: int,
    N: int,
    phi_target: float,
    seed: int,
    algorithm: str,
    lambda_f: float,
    n_bins: int = 3,
    beta: float = 0.0,
    confounding_strength: Optional[float] = None,
    enable_temporal_blacklist: bool = True,
    penalty_protected_indices: Optional[List[int]] = None,
    condition: Optional[str] = None,
) -> Dict[str, Any]:
    _set_seeds(seed)

    syn = generate_synthetic_dataset(
        archetype_id=archetype_id,
        N=N,
        phi_target=phi_target,
        n_bins=n_bins,
        seed=seed,
        confounding_strength=confounding_strength,
    )

    data = syn["data"]
    variable_names = syn["variable_names"]
    protected_indices = syn["protected_indices"]
    outcome_index = syn["outcome_index"]
    is_displacement = int(syn["archetype_id"]) == 4
    search_protected_indices = (
        list(penalty_protected_indices)
        if penalty_protected_indices is not None
        else list(protected_indices)
    )

    _validate_discrete_input(data)

    fairness_cache = build_fairness_cache(
        data=data,
        variable_names=variable_names,
        protected_indices=protected_indices,
        outcome_index=outcome_index,
        tau_c=5.0,
    )

    params = _algorithm_to_params(
        algorithm=algorithm,
        protected_indices=search_protected_indices,
        outcome_index=outcome_index,
        proxy_index=syn["proxy_index"],
        n_vars=len(variable_names),
        lambda_f=lambda_f,
        enable_temporal_blacklist=enable_temporal_blacklist,
        variable_names=variable_names,
        proxy_indices=syn.get("proxy_indices"),
    )
    score_func = _select_discrete_score_func(params["experiment_type"])

    ges_result = run_ges(
        data=data,
        score_func=score_func,
        node_names=variable_names,
        blacklist_matrix=params["blacklist_matrix"],
        whitelist_matrix=None,
        debug=False,
        experiment_type=params["experiment_type"],
        fairness_lambda=params["fairness_lambda"],
        fairness_tau_c=params["fairness_tau_c"],
        protected_attr_indices=params["protected_attr_indices"],
        outcome_index=params["outcome_index"],
        path_effect_method="distance",
    )

    learned_edges = _extract_learned_edges(
        algorithm=algorithm,
        ges_result=ges_result,
        data=data,
        variable_names=variable_names,
        protected_indices=search_protected_indices,
        outcome_index=outcome_index,
        lambda_f=lambda_f,
        beta=beta,
    )

    possible_parents_y: Optional[List[str]] = None
    graph_type = "dag"
    if algorithm != "fair_mec_full":
        try:
            cpdag = _is_cpdag_graph(ges_result["G"], len(variable_names))
            graph_type = "cpdag" if cpdag else "dag"
            if cpdag:
                possible_parents_y = _cpdag_neighbors_of_outcome(
                    graph=ges_result["G"],
                    variable_names=variable_names,
                    outcome_index=outcome_index,
                )
        except Exception:
            # Fallback to directed-parent metric if graph structure is unavailable.
            pass

    undirected_edges_for_metrics: List[Tuple[str, str]] = []
    if algorithm != "fair_mec_full" and graph_type == "cpdag":
        undirected_edges_for_metrics = list(ges_result.get("undirected_edges", []))

    if is_displacement:
        row = collect_displacement_metrics(
            learned_edges=learned_edges,
            syn=syn,
            algorithm=algorithm,
            lambda_f=lambda_f,
            seed=seed,
            N=N,
            n_bins=n_bins,
            condition=condition,
            penalty_protected_indices=search_protected_indices if params["fairness_lambda"] is not None else [],
            graph_type=graph_type,
            undirected_edges=undirected_edges_for_metrics,
        )
    else:
        row = collect_metrics(
            learned_edges=learned_edges,
            syn=syn,
            fairness_cache=fairness_cache,
            algorithm=algorithm,
            lambda_f=lambda_f,
            seed=seed,
            phi_target=phi_target,
            N=N,
            n_bins=n_bins,
            possible_parents_y=possible_parents_y,
            graph_type=graph_type,
            undirected_edges=undirected_edges_for_metrics,
        )
    row["enable_temporal_blacklist"] = bool(enable_temporal_blacklist)
    row["score_func"] = score_func
    return row



def save_results(rows: List[Dict[str, Any]], output_dir: str, config_snapshot: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "n_rows": int(len(rows)),
        "config": config_snapshot,
    }
    meta_path = os.path.join(output_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return csv_path
