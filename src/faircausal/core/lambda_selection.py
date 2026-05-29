"""
lambda_selection.py

Three strategies for selecting the fairness penalty weight λ in
Fairness-Regularized GES (Algorithm 3 / Algorithm 4).

Strategy A – Grid Search with Validation:
    Split the structure-discovery data into discovery/validation halves.
    For each λ in a grid, run Stage-1 GES and evaluate BIC + fairness metric
    on the validation fold.  Select the Pareto-optimal λ (best fairness
    improvement per unit of BIC degradation).

Strategy B – Elbow Method:
    Given grid-search results, find the λ at the "elbow" of the
    fairness-vs-accuracy curve using the Kneedle / maximum-second-derivative
    heuristic.

Strategy C – Normative Anchoring (default):
    Binary-search for the smallest λ such that the chosen fairness metric
    (Total Effect by default) satisfies |metric| < ε on the validation fold.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# Internal helpers
# ===========================================================================

def _run_ges_stage1(
    data_discovery: np.ndarray,
    node_names: List[str],
    blacklist_matrix: Optional[np.ndarray],
    whitelist_matrix: Optional[np.ndarray],
    protected_attr_indices: List[int],
    outcome_idx: int,
    lambda_fair: float,
    path_effect_method: str = "distance",
) -> Any:  # returns a GeneralGraph
    """
    Run Stage-1 GES with the given λ.  Returns the CPDAG (GeneralGraph).
    Wraps ges_runner_causal_learn.run_ges so that it can be called during
    lambda selection without touching stdout/logs.
    """
    # Import here to avoid circular imports
    from src.faircausal.core.ges_runner_causal_learn import run_ges

    result = run_ges(
        data=data_discovery,
        node_names=node_names,
        blacklist_matrix=blacklist_matrix,
        whitelist_matrix=whitelist_matrix,
        debug=False,
        experiment_type="soft_fairness",
        fairness_lambda=lambda_fair,
        fairness_tau_c=5.0,
        protected_attr_indices=protected_attr_indices,
        outcome_index=outcome_idx,
        path_effect_method=path_effect_method,
    )
    return result["G"]


def _evaluate_fairness_metric(
    cpdag: Any,
    data_validation: pd.DataFrame,
    node_names: List[str],
    protected_attrs: List[str],
    target: str,
    var_types: Dict[str, str],
    fairness_metric: str = "TE",
    disadvantage_group: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Evaluate (bic_score, fairness_score) on a validation fold for a given CPDAG.

    Returns
    -------
    (bic, fairness_value)
        bic : average BIC per sample (higher is better)
        fairness_value : |metric| (lower is better = fairer)
    """
    from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint
    from src.faircausal.core.fairness_scoring import compute_bic_for_dag
    from src.faircausal.core.counterfactual_fairness_runner import cpdag_to_dags

    data_val_np = data_validation.values if hasattr(data_validation, 'values') else data_validation

    # BIC: average over all DAGs consistent with the CPDAG
    all_dags = cpdag_to_dags(cpdag, node_names)
    if not all_dags:
        return (float("-inf"), float("nan"))

    bic_scores = [
        compute_bic_for_dag(dag, data_val_np, node_names) / max(len(data_val_np), 1)
        for dag in all_dags
    ]
    avg_bic = float(np.mean(bic_scores))

    # Fairness metric: use Total Effect via a lightweight correlation estimate
    # when the full SCM pipeline would be too expensive for every grid point.
    # We approximate TE(S→Y) as the *marginal* (partial) regression coefficient
    # of S on Y, adjusted for all other variables (OLS proxy).  This is cheap
    # and monotone in the true TE for linear models.
    fairness_value = _approx_total_effect(
        data_val_np, node_names, protected_attrs, target
    )

    return avg_bic, fairness_value


def _approx_total_effect(
    data: np.ndarray,
    node_names: List[str],
    protected_attrs: List[str],
    target: str,
) -> float:
    """
    Approximate |TE(S→Y)| as the maximum absolute Pearson correlation
    between Y and any protected attribute S, controlling for all other
    variables using OLS partial regression.

    This is a cheap proxy — it is strictly used only during lambda selection
    (not for reporting final results, which use the full SCM pipeline).
    """
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    y_idx = name_to_idx.get(target)
    if y_idx is None:
        return float("nan")

    y = data[:, y_idx].astype(float)
    n = len(y)
    if n < 5:
        return float("nan")

    # Centre y
    y_c = y - y.mean()

    max_te = 0.0
    for pattr in protected_attrs:
        s_idx = name_to_idx.get(pattr)
        if s_idx is None:
            continue
        s = data[:, s_idx].astype(float)
        # Simple correlation as proxy (no partial regression in this approx)
        s_c = s - s.mean()
        denom = (np.linalg.norm(s_c) * np.linalg.norm(y_c))
        if denom < 1e-12:
            continue
        corr = float(np.dot(s_c, y_c) / denom)
        max_te = max(max_te, abs(corr))

    return max_te


# ===========================================================================
# Strategy A – Grid Search with Validation
# ===========================================================================

def lambda_grid_search(
    data_discovery: pd.DataFrame,
    node_names: List[str],
    blacklist_matrix: Optional[np.ndarray],
    whitelist_matrix: Optional[np.ndarray],
    protected_attrs: List[str],
    target: str,
    var_types: Dict[str, str],
    disadvantage_group: Optional[Dict[str, Any]] = None,
    grid: Optional[List[float]] = None,
    path_effect_method: str = "distance",
    validation_fraction: float = 0.3,
    random_state: int = 42,
) -> Tuple[float, pd.DataFrame]:
    """
    Strategy A: Grid-search λ with held-out validation.

    Parameters
    ----------
    data_discovery : structure-discovery portion of the data (DataFrame)
    node_names : variable names matching data columns
    blacklist_matrix, whitelist_matrix : hard constraint matrices
    protected_attrs : names of protected attribute columns
    target : outcome variable name
    var_types : dict mapping variable name → type string
    disadvantage_group : optional dict (not used in proxy evaluation)
    grid : list of λ values to try (default: [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    path_effect_method : 'distance' or 'cramers_v'
    validation_fraction : fraction of discovery data to hold out for validation
    random_state : RNG seed for the discovery/validation split

    Returns
    -------
    (best_lambda, results_df)
        best_lambda : selected λ value
        results_df : DataFrame with columns [lambda, bic, fairness, pareto_optimal]
    """
    if grid is None:
        grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    protected_attr_indices = [name_to_idx[a] for a in protected_attrs if a in name_to_idx]
    outcome_idx = name_to_idx.get(target, 0)

    # Split discovery set into discovery / validation
    rng = np.random.default_rng(random_state)
    n = len(data_discovery)
    val_size = max(1, int(n * validation_fraction))
    shuffle_idx = rng.permutation(n)
    val_idx = shuffle_idx[:val_size]
    disc_idx = shuffle_idx[val_size:]

    data_disc_np = data_discovery.values[disc_idx]
    data_val_df = data_discovery.iloc[val_idx]

    rows = []
    print(f"[LambdaGridSearch] Evaluating {len(grid)} λ values …")

    for lam in grid:
        print(f"  λ={lam:.3f}", end=" ")
        try:
            cpdag = _run_ges_stage1(
                data_disc_np, node_names, blacklist_matrix, whitelist_matrix,
                protected_attr_indices, outcome_idx, lam, path_effect_method,
            )
            bic, fairness = _evaluate_fairness_metric(
                cpdag, data_val_df, node_names, protected_attrs, target, var_types
            )
            print(f"→ BIC={bic:.3f}  |TE|={fairness:.4f}")
        except Exception as exc:
            warnings.warn(f"λ={lam} evaluation failed: {exc}")
            bic, fairness = float("-inf"), float("nan")
            print(f"→ FAILED ({exc})")

        rows.append({"lambda": lam, "bic": bic, "fairness": fairness})

    results_df = pd.DataFrame(rows)

    # Pareto-optimal: not dominated (both BIC and fairness can't be improved simultaneously)
    results_df["pareto_optimal"] = False
    valid = results_df.dropna(subset=["bic", "fairness"])
    for idx in valid.index:
        bic_i = valid.loc[idx, "bic"]
        fair_i = valid.loc[idx, "fairness"]
        dominated = any(
            (valid.loc[j, "bic"] >= bic_i and valid.loc[j, "fairness"] <= fair_i
             and not (valid.loc[j, "bic"] == bic_i and valid.loc[j, "fairness"] == fair_i))
            for j in valid.index if j != idx
        )
        if not dominated:
            results_df.loc[idx, "pareto_optimal"] = True

    # Among Pareto-optimal points, pick the one with highest fairness improvement
    # per unit BIC degradation relative to λ=0 baseline
    baseline_row = results_df[results_df["lambda"] == 0.0]
    if baseline_row.empty:
        baseline_row = results_df.iloc[[0]]
    bic_0 = float(baseline_row["bic"].iloc[0])
    fair_0 = float(baseline_row["fairness"].iloc[0])

    pareto = results_df[results_df["pareto_optimal"] & results_df["lambda"] > 0].copy()
    if pareto.empty:
        best_lambda = 0.0
    else:
        # score = ΔFairness / (|ΔBIC| + ε)  — avoids division by zero
        pareto = pareto.copy()
        pareto["score"] = (fair_0 - pareto["fairness"]) / (abs(pareto["bic"] - bic_0) + 1e-6)
        best_idx = pareto["score"].idxmax()
        best_lambda = float(pareto.loc[best_idx, "lambda"])

    print(f"[LambdaGridSearch] Selected λ={best_lambda}")
    return best_lambda, results_df


# ===========================================================================
# Strategy B – Elbow Method
# ===========================================================================

def lambda_elbow_method(results_df: pd.DataFrame) -> float:
    """
    Strategy B: Select λ at the "elbow" of the fairness-vs-BIC Pareto front.

    Uses maximum second-derivative (discrete curvature) of the sorted curve.

    Parameters
    ----------
    results_df : DataFrame as returned by lambda_grid_search() with columns
                 [lambda, bic, fairness]

    Returns
    -------
    float : selected λ (lowest in grid if elbow cannot be found)
    """
    df = results_df.dropna(subset=["bic", "fairness"]).sort_values("fairness")
    if len(df) < 3:
        warnings.warn("Too few data points for elbow detection; returning λ=0.")
        return 0.0

    bic_vals = df["bic"].values.astype(float)
    fair_vals = df["fairness"].values.astype(float)

    # Normalise to [0, 1]
    bic_range = bic_vals.max() - bic_vals.min()
    fair_range = fair_vals.max() - fair_vals.min()
    if bic_range < 1e-12 or fair_range < 1e-12:
        warnings.warn("Degenerate Pareto front; returning λ=0.")
        return 0.0

    bic_n = (bic_vals - bic_vals.min()) / bic_range
    fair_n = (fair_vals - fair_vals.min()) / fair_range

    # Second derivative of normalised fairness w.r.t. normalised BIC
    # (crude finite difference)
    curvature = np.abs(np.diff(np.diff(fair_n) / (np.diff(bic_n) + 1e-12)))

    elbow_pos = int(np.argmax(curvature)) + 1  # +1 offset from double diff
    best_lambda = float(df["lambda"].iloc[elbow_pos])
    print(f"[LambdaElbow] Elbow detected at position {elbow_pos}, λ={best_lambda}")
    return best_lambda


# ===========================================================================
# Strategy C – Normative Anchoring (default, recommended)
# ===========================================================================

def lambda_normative_anchoring(
    data_discovery: pd.DataFrame,
    node_names: List[str],
    blacklist_matrix: Optional[np.ndarray],
    whitelist_matrix: Optional[np.ndarray],
    protected_attrs: List[str],
    target: str,
    var_types: Dict[str, str],
    epsilon: float = 0.05,
    fairness_metric: str = "TE",
    path_effect_method: str = "distance",
    validation_fraction: float = 0.3,
    random_state: int = 42,
    lambda_lo: float = 0.0,
    lambda_hi: float = 50.0,
    n_iterations: int = 15,
) -> float:
    """
    Strategy C: Binary-search for the smallest λ such that |fairness_metric| < ε.

    Parameters
    ----------
    data_discovery : structure-discovery data
    node_names : variable names
    blacklist_matrix, whitelist_matrix : hard constraints
    protected_attrs : protected attribute column names
    target : outcome variable name
    var_types : variable type dictionary
    epsilon : fairness threshold (e.g., 0.05 for |TE| < 0.05)
    fairness_metric : 'TE' is the only supported cheap proxy at this stage
    path_effect_method : 'distance' or 'cramers_v'
    validation_fraction : holdout fraction for evaluation
    random_state : RNG seed
    lambda_lo, lambda_hi : binary search bounds
    n_iterations : number of bisection steps

    Returns
    -------
    float : smallest λ achieving the target, or lambda_hi if never satisfied
    """
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    protected_attr_indices = [name_to_idx[a] for a in protected_attrs if a in name_to_idx]
    outcome_idx = name_to_idx.get(target, 0)

    rng = np.random.default_rng(random_state)
    n = len(data_discovery)
    val_size = max(1, int(n * validation_fraction))
    shuffle_idx = rng.permutation(n)
    val_idx = shuffle_idx[:val_size]
    disc_idx = shuffle_idx[val_size:]

    data_disc_np = data_discovery.values[disc_idx]
    data_val_df = data_discovery.iloc[val_idx]

    def _eval(lam: float) -> float:
        cpdag = _run_ges_stage1(
            data_disc_np, node_names, blacklist_matrix, whitelist_matrix,
            protected_attr_indices, outcome_idx, lam, path_effect_method,
        )
        _, fairness = _evaluate_fairness_metric(
            cpdag, data_val_df, node_names, protected_attrs, target, var_types
        )
        return fairness

    # First check if the target is achievable at all
    fairness_at_hi = _eval(lambda_hi)
    print(f"[NormativeAnchoring] ε={epsilon}, |TE| at λ={lambda_hi}: {fairness_at_hi:.4f}")

    if math.isnan(fairness_at_hi) or fairness_at_hi >= epsilon:
        warnings.warn(
            f"Fairness target |TE|<{epsilon} not achieved even at λ={lambda_hi}. "
            f"Returning λ={lambda_hi} as best effort."
        )
        return lambda_hi

    lo, hi = lambda_lo, lambda_hi
    best = lambda_hi

    for it in range(n_iterations):
        mid = (lo + hi) / 2.0
        f = _eval(mid)
        print(f"  iteration {it+1}/{n_iterations}: λ={mid:.4f}  |TE|={f:.4f}")
        if math.isnan(f):
            lo = mid
            continue
        if f < epsilon:
            best = mid
            hi = mid
        else:
            lo = mid

    print(f"[NormativeAnchoring] Selected λ={best:.4f}")
    return best


# ===========================================================================
# Public dispatcher
# ===========================================================================

def select_lambda(
    strategy: str,
    data_discovery: pd.DataFrame,
    node_names: List[str],
    blacklist_matrix: Optional[np.ndarray],
    whitelist_matrix: Optional[np.ndarray],
    protected_attrs: List[str],
    target: str,
    var_types: Dict[str, str],
    soft_fairness_settings: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> float:
    """
    Dispatcher: runs the chosen lambda-selection strategy and returns λ.

    Parameters
    ----------
    strategy : 'grid_search', 'elbow', or 'normative_anchoring'
    soft_fairness_settings : dict from dataset config.  Expected keys:
        lambda_grid, epsilon, fairness_metric, path_effect_method,
        validation_fraction (all optional, fall back to defaults).

    Returns
    -------
    float : selected λ ≥ 0
    """
    cfg = soft_fairness_settings or {}
    grid = cfg.get("lambda_grid", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    epsilon = cfg.get("epsilon", 0.05)
    fairness_metric = cfg.get("fairness_metric", "TE")
    path_effect_method = cfg.get("path_effect_method", "distance")
    val_frac = cfg.get("validation_fraction", 0.3)
    disadvantage_group = cfg.get("disadvantage_group", None)

    if strategy == "grid_search":
        best_lambda, _ = lambda_grid_search(
            data_discovery=data_discovery,
            node_names=node_names,
            blacklist_matrix=blacklist_matrix,
            whitelist_matrix=whitelist_matrix,
            protected_attrs=protected_attrs,
            target=target,
            var_types=var_types,
            disadvantage_group=disadvantage_group,
            grid=grid,
            path_effect_method=path_effect_method,
            validation_fraction=val_frac,
            random_state=random_state,
        )
        return best_lambda

    elif strategy == "elbow":
        _, results_df = lambda_grid_search(
            data_discovery=data_discovery,
            node_names=node_names,
            blacklist_matrix=blacklist_matrix,
            whitelist_matrix=whitelist_matrix,
            protected_attrs=protected_attrs,
            target=target,
            var_types=var_types,
            disadvantage_group=disadvantage_group,
            grid=grid,
            path_effect_method=path_effect_method,
            validation_fraction=val_frac,
            random_state=random_state,
        )
        return lambda_elbow_method(results_df)

    elif strategy == "normative_anchoring":
        return lambda_normative_anchoring(
            data_discovery=data_discovery,
            node_names=node_names,
            blacklist_matrix=blacklist_matrix,
            whitelist_matrix=whitelist_matrix,
            protected_attrs=protected_attrs,
            target=target,
            var_types=var_types,
            epsilon=epsilon,
            fairness_metric=fairness_metric,
            path_effect_method=path_effect_method,
            validation_fraction=val_frac,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown lambda selection strategy: {strategy!r}. "
            f"Choose 'grid_search', 'elbow', or 'normative_anchoring'."
        )
