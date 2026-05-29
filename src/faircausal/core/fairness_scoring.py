"""
fairness_scoring.py

Fairness scoring utilities for ProxyFair.

Implements the two-phase ProxyFair penalty framework:

Phase 1 (score-equivalent GES search) — precomputed, symmetric:
    - Proxy-excess fraction aggregation π^Σ(i,j) from
        compute_edge_proxy_fractions(...) (paper Eq. for proxy-excess fraction)
    - Outcome relevance ρ_adj(i) from compute_outcome_relevance(...)
    - Directed penalty ψ(i→j) = π^Σ(i,j) * ρ_adj(j)
    - Symmetric penalty Ψ_sym(i,j) = 0.5 * (ψ(i→j) + ψ(j→i))
    - Fair local score uses λ_f * N scaling with Ψ_sym to preserve score equivalence

Phase 2 (Fair-DAG extraction from CPDAG) — directed, graph-aware:
    - Uses directed penalty terms to rank DAG extensions after CPDAG search
    - Keeps fairness selection directional while Phase 1 remains score-equivalent

Legacy pre-ProxyFair helpers are retained for backward compatibility with older
experiments. They are grouped below and are not part of the paper's primary
π^Σ-based algorithm path.

Graph encoding (GeneralGraph / GES_fair convention):
  G.graph[i, j] == TAIL (-1) and G.graph[j, i] == ARROW (1)  =>  i → j
  G.graph[i, j] == TAIL (-1) and G.graph[j, i] == TAIL (-1)  =>  i — j (undirected)
  G.graph[i, j] == NULL (0)                                    =>  no edge
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Endpoint constants (mirrors causallearn.graph.Endpoint)
# ---------------------------------------------------------------------------
_TAIL = -1   # Endpoint.TAIL.value
_ARROW = 1   # Endpoint.ARROW.value
_NULL = 0    # Endpoint.NULL.value


# ===========================================================================
# Section 1 – Discrete information-theoretic utilities
# ===========================================================================

def discrete_entropy(col: np.ndarray) -> float:
    """
    Compute the Shannon entropy H(X) of a 1-D categorical array.

    H(X) = -Σ p(x) log p(x)   (nats)

    Returns 0.0 for a degenerate (constant) variable.
    """
    col = np.asarray(col).ravel()
    _, counts = np.unique(col, return_counts=True)
    n = col.shape[0]
    if n == 0:
        return 0.0
    probs = counts / n
    # log(0) guard: multiply log by (probs > 0) implicitly
    return -float(np.sum(probs * np.log(probs + 1e-300)))


def discrete_mutual_information(x: np.ndarray, s: np.ndarray) -> float:
    """
    Estimate the mutual information MI(X; S) from samples via contingency table.

    MI(X; S) = Σ_{x,s} p(x,s) log [ p(x,s) / (p(x) p(s)) ]  (nats)

    Uses maximum-likelihood estimates (add-0 smoothing).  For small sample/
    category ratios consider the returned value unreliable — a warning is
    emitted when any cell has < 5 expected counts.
    """
    x = np.asarray(x).ravel()
    s = np.asarray(s).ravel()
    n = x.shape[0]
    if n == 0:
        return 0.0

    x_vals, x_inv = np.unique(x, return_inverse=True)
    s_vals, s_inv = np.unique(s, return_inverse=True)
    Kx, Ks = len(x_vals), len(s_vals)

    # Build joint contingency table
    joint = np.zeros((Kx, Ks), dtype=np.float64)
    np.add.at(joint, (x_inv, s_inv), 1)

    # Warn on sparse tables
    expected_min = n / (Kx * Ks)
    if expected_min < 5:
        warnings.warn(
            f"MI estimate unreliable: expected cell count {expected_min:.1f} < 5 "
            f"(n={n}, Kx={Kx}, Ks={Ks}).  Consider merging categories.",
            stacklevel=2,
        )

    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)  # (Kx, 1)
    py = pxy.sum(axis=0, keepdims=True)  # (1, Ks)

    # Only sum where joint probability > 0
    mask = pxy > 0
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py + 1e-300)[mask])))
    return max(0.0, mi)  # numerical noise guard


def discrete_conditional_mutual_information(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> float:
    """
    Estimate the conditional mutual information I(X; Y | Z) from samples via
    a 3-way contingency table.

    I(X; Y | Z) = Σ_{x,y,z} p(x,y,z) log[ p(x,y|z) / (p(x|z)·p(y|z)) ]
                = Σ_z p(z) · MI(X; Y | Z=z)   (nats)

    Uses maximum-likelihood estimates.  Returns a non-negative float;
    small negative values from floating-point noise are clipped to 0.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    n = x.shape[0]
    if n == 0:
        return 0.0

    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    z_vals, z_inv = np.unique(z, return_inverse=True)
    Kx, Ky, Kz = len(x_vals), len(y_vals), len(z_vals)

    # Build 3-way joint table  shape: (Kx, Ky, Kz)
    joint3 = np.zeros((Kx, Ky, Kz), dtype=np.float64)
    np.add.at(joint3, (x_inv, y_inv, z_inv), 1)
    joint3 /= n

    # Marginal over z: p(x,z) and p(y,z)
    pxz = joint3.sum(axis=1)   # (Kx, Kz)
    pyz = joint3.sum(axis=0)   # (Ky, Kz)
    pz  = joint3.sum(axis=(0, 1))  # (Kz,)

    cmi = 0.0
    for k in range(Kz):
        if pz[k] < 1e-300:
            continue
        # p(x,y|z=k) and marginals p(x|z=k), p(y|z=k)
        pxy_z = joint3[:, :, k] / pz[k]            # (Kx, Ky)
        px_z  = pxz[:, k] / pz[k]                  # (Kx,)
        py_z  = pyz[:, k] / pz[k]                  # (Ky,)
        denom = px_z[:, None] * py_z[None, :]       # (Kx, Ky)
        mask = pxy_z > 0
        cmi += pz[k] * float(
            np.sum(pxy_z[mask] * np.log(pxy_z[mask] / (denom + 1e-300)[mask]))
        )
    return max(0.0, cmi)


def compute_proxy_strength_matrix(
    data: np.ndarray,
    node_names: List[str],
    protected_attr_indices: List[int],
) -> Dict[Tuple[int, int], float]:
    """
    Precompute proxy strength φ(X, S) = MI(X; S) / min(H(X), H(S)) for all
    (variable, protected_attr) pairs.  This is the normalised mutual information
    (NMI) with min-normalisation, yielding φ ∈ [0, 1] regardless of the relative
    cardinalities of X and S (Definition 1 in the paper).

    Returns a dict mapping (variable_index, protected_attr_index) → float in [0, 1].
    When min(H(X), H(S)) == 0 (constant variable), φ is defined as 0.
    """
    n_vars = data.shape[1]
    proxy_strength: Dict[Tuple[int, int], float] = {}

    # Cache entropies to avoid recomputing them for every pair
    entropies = [discrete_entropy(data[:, k]) for k in range(n_vars)]

    for x_idx in range(n_vars):
        hx = entropies[x_idx]
        for s_idx in protected_attr_indices:
            if x_idx == s_idx:
                # A protected attribute is trivially its own proxy (strength = 1)
                proxy_strength[(x_idx, s_idx)] = 1.0
                continue
            hs = entropies[s_idx]
            denom = min(hx, hs)
            if denom < 1e-12:
                proxy_strength[(x_idx, s_idx)] = 0.0
            else:
                mi = discrete_mutual_information(data[:, x_idx], data[:, s_idx])
                proxy_strength[(x_idx, s_idx)] = min(1.0, mi / denom)

    return proxy_strength


# ===========================================================================
# Legacy helpers (not part of the paper's primary π^Σ ProxyFair path)
# ===========================================================================

# ===========================================================================
# Section 2 – PathEffect helpers (graph reachability and association strength)
# ===========================================================================

def _conservative_neighbors(graph_matrix: np.ndarray, node: int) -> List[int]:
    """
    Return all nodes reachable from `node` in one step under conservative CPDAG handling:
      - directed edge node → v  (graph[node,v]=TAIL, graph[v,node]=ARROW)
      - undirected edge node — v (graph[node,v]=TAIL, graph[v,node]=TAIL)
    Both are considered traversable in the forward direction.
    """
    n = graph_matrix.shape[0]
    neighbors = []
    for v in range(n):
        if v == node:
            continue
        # Any non-NULL connection from node to v is conservative-traversable
        if graph_matrix[node, v] != _NULL:
            neighbors.append(v)
    return neighbors


def find_shortest_reachable_distance(
    graph_matrix: np.ndarray, source: int, target: int
) -> int:
    """
    BFS from `source` to `target` treating undirected edges as bidirectional
    (conservative CPDAG handling).

    Returns the shortest path length (number of edges), or -1 if unreachable.
    """
    if source == target:
        return 0

    visited = set([source])
    queue = deque([(source, 0)])

    while queue:
        node, dist = queue.popleft()
        for neighbor in _conservative_neighbors(graph_matrix, node):
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return -1  # unreachable


def path_effect_distance(
    graph_matrix: np.ndarray, z_idx: int, y_idx: int
) -> float:
    """
    PathEffect variant 1 (recommended): distance-based damping.

      PathEffect(Z, Y) = 1.0              if Z == Y
                       = 1 / d(Z, Y)      if Z can reach Y  (d = shortest BFS path)
                       = 0.0              otherwise

    Conservative CPDAG handling: undirected edges are traversable in either
    direction, providing an over-approximation of reachability.
    """
    if z_idx == y_idx:
        return 1.0
    d = find_shortest_reachable_distance(graph_matrix, z_idx, y_idx)
    if d < 0:
        return 0.0
    return 1.0 / d


def cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cramer's V association measure between two categorical arrays x and y.

    V = sqrt(χ² / (n · min(Kx-1, Ky-1)))

    Returns 0.0 when the table is degenerate (constant column or row).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = x.shape[0]
    if n == 0:
        return 0.0

    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    Kx, Ky = len(x_vals), len(y_vals)

    if Kx <= 1 or Ky <= 1:
        return 0.0

    # Contingency table → χ²
    table = np.zeros((Kx, Ky), dtype=np.float64)
    np.add.at(table, (x_inv, y_inv), 1)

    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum * col_sum / n

    # Avoid division by zero in cells where expected == 0
    mask = expected > 0
    chi2 = float(np.sum((table[mask] - expected[mask]) ** 2 / expected[mask]))

    denom = n * (min(Kx, Ky) - 1)
    if denom <= 0:
        return 0.0
    return min(1.0, math.sqrt(chi2 / denom))


def _find_all_paths_conservative(
    graph_matrix: np.ndarray, source: int, target: int, max_depth: int = 20
) -> List[List[int]]:
    """
    BFS-based enumeration of all simple paths from source to target under
    conservative CPDAG handling (both directed and undirected edges traversed
    forward).  Capped at `max_depth` to avoid infinite loops on large graphs.
    """
    if source == target:
        return [[source]]

    paths: List[List[int]] = []
    # Each queue element is (current_node, path_so_far)
    queue: deque = deque([(source, [source])])

    while queue:
        node, path = queue.popleft()
        if len(path) > max_depth:
            continue
        for neighbor in _conservative_neighbors(graph_matrix, node):
            if neighbor in path:
                continue  # no cycles
            new_path = path + [neighbor]
            if neighbor == target:
                paths.append(new_path)
            else:
                queue.append((neighbor, new_path))

    return paths


def path_effect_cramers_v(
    graph_matrix: np.ndarray,
    z_idx: int,
    y_idx: int,
    data: np.ndarray,
) -> float:
    """
    PathEffect variant 2: product of Cramer's V along the strongest path from Z to Y.

      PathEffect(Z, Y) = 1.0                                           if Z == Y
                       = max_p Π_{(u,v)∈p} CramersV(data[:,u], data[:,v])  if path exists
                       = 0.0                                           otherwise

    Conservative CPDAG handling applies.
    """
    if z_idx == y_idx:
        return 1.0

    all_paths = _find_all_paths_conservative(graph_matrix, z_idx, y_idx)
    if not all_paths:
        return 0.0

    best = 0.0
    for path in all_paths:
        product = 1.0
        for u, v in zip(path[:-1], path[1:]):
            product *= cramers_v(data[:, u], data[:, v])
        best = max(best, product)

    return best


# ===========================================================================
# Section 3 – Combine proxy strength and path effect into Φ
# ===========================================================================

def compute_fairness_penalty(
    graph_matrix: np.ndarray,
    x_idx: int,
    z_idx: int,
    proxy_strength: Dict[Tuple[int, int], float],
    protected_attr_indices: List[int],
    outcome_idx: int,
    data: Optional[np.ndarray] = None,
    path_effect_method: str = "distance",
) -> float:
    """
    Compute the local fairness penalty Φ for a candidate edge X → Z.

      Φ(X → Z; G, D) = Σ_{s ∈ S} ProxyStrength(X, s) · PathEffect(Z, Y; G)

    The PathEffect is evaluated from Z to Y (outcome), measuring how strongly
    the added edge propagates its effect toward the outcome.

    Parameters
    ----------
    graph_matrix : G.graph from a GeneralGraph (N×N int array)
    x_idx : source variable index (X)
    z_idx : destination variable index (Z)
    proxy_strength : precomputed dict from compute_proxy_strength_matrix()
    protected_attr_indices : list of protected attribute column indices
    outcome_idx : column index of the outcome variable Y
    data : original data array — required when path_effect_method='cramers_v'
    path_effect_method : 'distance' (default) or 'cramers_v'

    Returns
    -------
    float : Φ ≥ 0
    """
    if path_effect_method == "distance":
        pe = path_effect_distance(graph_matrix, z_idx, outcome_idx)
    elif path_effect_method == "cramers_v":
        if data is None:
            raise ValueError("data must be provided when path_effect_method='cramers_v'")
        pe = path_effect_cramers_v(graph_matrix, z_idx, outcome_idx, data)
    else:
        raise ValueError(f"Unknown path_effect_method: {path_effect_method!r}. "
                         f"Choose 'distance' or 'cramers_v'.")

    if pe == 0.0:
        return 0.0

    total = 0.0
    for s_idx in protected_attr_indices:
        ps = proxy_strength.get((x_idx, s_idx), 0.0)
        total += ps * pe

    return total


# ===========================================================================
# Section 3b – Phase 1 precomputed symmetric penalty (paper Eq. 2-5)
# ===========================================================================

def compute_edge_proxy_fractions(
    data: np.ndarray,
    protected_attr_indices: List[int],
    tau_c: float = 5.0,
    min_group_size: int = 10,
    var_names: Optional[List[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Compute edge-level proxy fraction π^Σ(i, j) for all variable pairs.

    For each unordered pair (i, j), compute:
      - marginal association: NMI(V_i, V_j)
      - conditional association wrt each protected attribute S_k:
          weighted mean of NMI(V_i, V_j) within groups of S_k
      - per-protected proxy fraction:
          π_k(i, j) = max(0, (marginal - conditional_k) / marginal)
      - aggregate:
          π^Σ(i, j) = Σ_k π_k(i, j)

    A noise floor τ = tau_c / (N - 2) is applied to the marginal NMI.
    If marginal < τ, π^Σ(i, j) is set to 0.

    Returns
    -------
    np.ndarray
        Symmetric matrix pi_sigma with shape (p, p).
    """
    n_samples, n_vars = data.shape
    pi_sigma = np.zeros((n_vars, n_vars), dtype=float)

    if n_samples <= 2:
        return pi_sigma

    tau = float(tau_c) / float(n_samples - 2)

    # Precompute S_k group slices once and log dropped observations per split.
    split_groups: Dict[int, List[np.ndarray]] = {}
    for s_idx in protected_attr_indices:
        groups: List[np.ndarray] = []
        dropped_obs = 0
        s_col = data[:, s_idx]
        for s_val in np.unique(s_col):
            idxs = np.where(s_col == s_val)[0]
            if idxs.size < min_group_size:
                dropped_obs += int(idxs.size)
                continue
            groups.append(idxs)
        split_groups[s_idx] = groups
        if logger is not None:
            s_name = var_names[s_idx] if var_names is not None and s_idx < len(var_names) else str(s_idx)
            logger(
                f"[Fairness] Conditional NMI split for {s_name}: "
                f"dropped={dropped_obs}/{n_samples}, retained={n_samples - dropped_obs}/{n_samples} "
                f"(min_group_size={min_group_size})"
            )

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            marginal_nmi = float(normalized_mutual_info_score(data[:, i], data[:, j]))
            if marginal_nmi < tau or marginal_nmi <= 1e-12:
                continue

            pi_sum = 0.0
            for s_idx in protected_attr_indices:
                groups = split_groups.get(s_idx, [])
                cond_nmi = 0.0
                for idxs in groups:
                    weight = float(idxs.size) / float(n_samples)
                    cond_nmi += weight * float(
                        normalized_mutual_info_score(data[idxs, i], data[idxs, j])
                    )
                pi_k = max(0.0, (marginal_nmi - cond_nmi) / marginal_nmi)
                pi_sum += pi_k

            pi_sigma[i, j] = pi_sum
            pi_sigma[j, i] = pi_sum

    assert np.allclose(pi_sigma, pi_sigma.T, atol=1e-10), "pi_sigma must be symmetric"
    return pi_sigma


def compute_outcome_relevance(
    data: np.ndarray,
    outcome_index: int,
) -> Dict[int, float]:
    """
    Simplified Stage-1 outcome relevance.

    rho_adj(i) = 1 for the outcome node, otherwise NMI(V_i, Y).
    """
    n_vars = data.shape[1]
    rho_adj: Dict[int, float] = {}
    for i in range(n_vars):
        if i == outcome_index:
            rho_adj[i] = 1.0
        else:
            rho_adj[i] = float(normalized_mutual_info_score(data[:, i], data[:, outcome_index]))
    return rho_adj


def compute_directed_penalty(
    pi_sigma: np.ndarray,
    rho_adj: Dict[int, float],
    source_idx: int,
    target_idx: int,
) -> float:
    """Compute directed edge penalty ψ(i→j) = π^Σ(i,j) * ρ_adj(j)."""
    return float(pi_sigma[source_idx, target_idx]) * float(rho_adj.get(target_idx, 0.0))


def compute_symmetric_penalty_from_pi(
    pi_sigma: np.ndarray,
    rho_adj: Dict[int, float],
) -> Dict[Tuple[int, int], float]:
    """
    Build symmetric edge penalty lookup from π^Σ and ρ_adj.

    Ψ_sym(i, j) = 0.5 * (ψ(i→j) + ψ(j→i)).
    """
    n_vars = pi_sigma.shape[0]
    psi: Dict[Tuple[int, int], float] = {}
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            psi_ij = compute_directed_penalty(pi_sigma, rho_adj, i, j)
            psi_ji = compute_directed_penalty(pi_sigma, rho_adj, j, i)
            psi[(i, j)] = 0.5 * (psi_ij + psi_ji)
    return psi


def compute_proxy_load(
    proxy_strength: Dict[Tuple[int, int], float],
    n_vars: int,
    protected_attr_indices: List[int],
) -> Dict[int, float]:
    """
    Legacy helper kept for backward compatibility with older experiments.

    Compute the proxy load φ*(V_j) = max_{S_k ∈ S} φ(V_j, S_k) for each variable.

    A variable is treated as a proxy if it strongly encodes *any* single protected
    attribute (anti-subordination stance, Eq. 2 in the paper).

    Returns a dict mapping variable_index → φ* ∈ [0, 1].
    """
    proxy_load: Dict[int, float] = {}
    for j in range(n_vars):
        proxy_load[j] = max(
            (proxy_strength.get((j, s_idx), 0.0) for s_idx in protected_attr_indices),
            default=0.0,
        )
    return proxy_load


def compute_static_outcome_relevance(
    data: np.ndarray,
    outcome_index: int,
) -> Dict[int, float]:
    """
    Legacy helper kept for backward compatibility with older experiments.

    Compute the static outcome relevance ρ_static(V_i) for every variable (Eq. 3).

    ρ_static(V_i) = 1                       if V_i == Y
                  = NMI(V_i, Y)              otherwise

    where NMI uses min-normalisation: MI(V_i, Y) / min(H(V_i), H(Y)), consistent
    with the proxy strength definition (Definition 1).  The result is graph-
    independent and computed once before the GES search begins.

    Returns a dict mapping variable_index → ρ_static ∈ [0, 1].
    """
    n_vars = data.shape[1]
    hy = discrete_entropy(data[:, outcome_index])
    relevance: Dict[int, float] = {}
    for i in range(n_vars):
        if i == outcome_index:
            relevance[i] = 1.0
        else:
            hi = discrete_entropy(data[:, i])
            denom = min(hi, hy)
            if denom < 1e-12:
                relevance[i] = 0.0
            else:
                mi = discrete_mutual_information(data[:, i], data[:, outcome_index])
                relevance[i] = min(1.0, mi / denom)
    return relevance


def compute_proxy_excess_association(
    data: np.ndarray,
    outcome_index: int,
    protected_attr_indices: List[int],
) -> Dict[int, float]:
    """
    Compute the proxy-excess association δ*(V_j) for every variable (proxy-excess
    correction described alongside Eq. 3 in the paper).

    For each protected attribute S_k, the drop in normalised MI when conditioning on S_k
    quantifies how much of V_j's association with Y was *mediated* by S_k (i.e. V_j is
    acting as a proxy for S_k):

        δ(V_j, Y, S_k) = max(0, NMI(V_j; Y) − cNMI(V_j; Y | S_k))

    where cNMI uses the same min-normalisation as ρ_static (Eq. 3):

        cNMI(V_j; Y | S_k) = I(V_j; Y | S_k) / min(H(V_j), H(Y))

    A positive δ means the association drops once S_k is controlled for, confirming
    that S_k mediates the V_j–Y relationship.  Negative values (suppression effects)
    are clipped to 0 so they do not reduce the penalty below the marginal baseline.

    δ*(V_j) = max_{S_k ∈ S} δ(V_j, Y, S_k)

    Returns a dict mapping variable_index → δ* ∈ [0, 1].
    """
    n_vars = data.shape[1]
    hy = discrete_entropy(data[:, outcome_index])
    delta_star: Dict[int, float] = {}

    for j in range(n_vars):
        hj = discrete_entropy(data[:, j])
        denom = min(hj, hy)

        # Marginal NMI(V_j; Y) — replicates ρ_static[j] computation
        if denom < 1e-12:
            nmi_marginal = 0.0
        else:
            nmi_marginal = min(
                1.0,
                discrete_mutual_information(data[:, j], data[:, outcome_index]) / denom,
            )

        best_delta = 0.0
        for s_idx in protected_attr_indices:
            if denom < 1e-12:
                # V_j is constant — cNMI is also 0, no excess
                continue
            cmi = discrete_conditional_mutual_information(
                data[:, j], data[:, outcome_index], data[:, s_idx]
            )
            cnmi = min(1.0, cmi / denom)
            delta_k = max(0.0, nmi_marginal - cnmi)
            if delta_k > best_delta:
                best_delta = delta_k

        delta_star[j] = best_delta

    return delta_star


def compute_adjusted_outcome_relevance(
    rho_static: Dict[int, float],
    delta_star: Dict[int, float],
    alpha: float,
) -> Dict[int, float]:
    """
    Compute the proxy-excess-adjusted outcome relevance ρ_adj(V_i).

        ρ_adj(V_i) = min(1, ρ_static(V_i) + α · δ*(V_i))

    α ∈ [0, 1] controls how much the proxy-excess correction boosts the penalty
    for variables that are strong proxies of S.  When α = 0, ρ_adj == ρ_static
    and behaviour is identical to the uncorrected algorithm.

    Returns a dict mapping variable_index → ρ_adj ∈ [0, 1].
    """
    return {
        i: min(1.0, rho_static.get(i, 0.0) + alpha * delta_star.get(i, 0.0))
        for i in rho_static
    }


def compute_symmetric_penalty_matrix(
    proxy_load: Dict[int, float],
    outcome_relevance: Dict[int, float],
    n_vars: int,
) -> Dict[Tuple[int, int], float]:
    """
    Legacy helper kept for backward compatibility with older experiments.

    Compute the symmetric edge penalty matrix Ψ_sym for all variable pairs (Eq. 4-5).

    For each ordered pair (i, j):
        ψ(V_j → V_i) = ρ_static(V_i) · φ*(V_j)   [j is parent, i is child]
        ψ(V_i → V_j) = ρ_static(V_j) · φ*(V_i)   [i is parent, j is child]
        Ψ_sym(V_i, V_j) = mean(ψ(V_j → V_i), ψ(V_i → V_j))

    Ψ_sym(i, j) == Ψ_sym(j, i) by construction.  This preserves score equivalence
    in GES: any two DAGs in the same MEC incur the same total symmetric penalty,
    just attributed to different child nodes.

    Returns a dict mapping (i, j) → Ψ_sym ∈ [0, 1] for all i ≠ j.
    """
    psi: Dict[Tuple[int, int], float] = {}
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            # ψ(V_j → V_i): j is parent, i is child
            psi_j_to_i = outcome_relevance.get(i, 0.0) * proxy_load.get(j, 0.0)
            # ψ(V_i → V_j): i is parent, j is child
            psi_i_to_j = outcome_relevance.get(j, 0.0) * proxy_load.get(i, 0.0)
            psi[(i, j)] = 0.5 * (psi_j_to_i + psi_i_to_j)
    return psi


# ===========================================================================
# Section 4 – Score integration helper
# ===========================================================================

def fairness_adjusted_delta(
    bic_delta: float,
    fairness_penalty: float,
    n_samples: int,
    lambda_fair: float,
) -> float:
    """
    Adjust the BIC score delta by the Phase 1 symmetric fairness penalty (Eq. 6).

            ΔFS = ΔBIC + λ_f · N · Ψ_sym

        The penalty scales as O(N), matching the likelihood signal order so that
        λ_f remains interpretable and comparable across datasets.

    Note: in the GES inner loops, lower chscore = better (GES minimises the
    *negative* BIC).  Adding a positive quantity makes unfair edges look worse,
    which is the correct direction.  For edge *removal* in the backward phase,
    the caller negates the penalty before calling this function.

    Parameters
    ----------
    bic_delta     : raw ΔBIC returned by insert_changed_score / delete_changed_score
    fairness_penalty : Ψ_sym(i, j) ≥ 0 from compute_symmetric_penalty_matrix()
    n_samples     : dataset size N
    lambda_fair   : fairness weight λ_f ≥ 0

    Returns
    -------
    float : adjusted score delta
    """
    if lambda_fair == 0.0 or fairness_penalty == 0.0:
        return bic_delta
    return bic_delta + lambda_fair * n_samples * fairness_penalty


# ===========================================================================
# Section 5 – BIC score for a full DAG (used in Stage 2 MEC selection)
# ===========================================================================

def compute_bic_for_dag(
    dag_edges: List[Tuple[str, str]],
    data: np.ndarray,
    node_names: List[str],
) -> float:
    """
    Compute the BIC score of a DAG given as a list of (source_name, target_name) tuples.

    Uses the same categorical BIC formulation as OptimizedBICScore in GES_fair.py
    (multinomial log-likelihood with log(N)/2 penalty per parameter).

    Parameters
    ----------
    dag_edges : list of (src_name, dst_name) tuples
    data : full data array, shape (n_samples, n_features)
    node_names : list mapping column position → variable name

    Returns
    -------
    float : total BIC score (higher is better, negative values are normal)
    """
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    n_samples, n_vars = data.shape

    # Build parent set for each variable
    parents: Dict[int, List[int]] = {i: [] for i in range(n_vars)}
    for src_name, dst_name in dag_edges:
        if src_name in name_to_idx and dst_name in name_to_idx:
            parents[name_to_idx[dst_name]].append(name_to_idx[src_name])

    total_bic = 0.0
    for i in range(n_vars):
        total_bic += _local_categorical_bic(data, i, parents[i], n_samples)

    return total_bic


def _local_categorical_bic(
    data: np.ndarray,
    i: int,
    pai: List[int],
    n_samples: int,
) -> float:
    """Local BIC term for categorical variable i given parents pai."""
    target = data[:, i]
    t_vals = np.unique(target)
    n_categories = len(t_vals)
    val_to_idx = {v: k for k, v in enumerate(t_vals)}

    if not pai:
        counts = np.zeros(n_categories)
        for v in target:
            counts[val_to_idx[v]] += 1
        nz = counts[counts > 0]
        log_lik = float(np.sum(nz * np.log(nz))) - n_samples * math.log(n_samples)
        penalty = -0.5 * (n_categories - 1) * math.log(n_samples)
        return log_lik + penalty

    from collections import defaultdict
    config_counts: dict = defaultdict(lambda: np.zeros(n_categories))
    for row in range(n_samples):
        parent_config = tuple(data[row, p] for p in pai)
        config_counts[parent_config][val_to_idx[target[row]]] += 1

    log_lik = 0.0
    n_parent_configs = len(config_counts)
    for counts in config_counts.values():
        s = counts.sum()
        if s > 0:
            nz = counts[counts > 0]
            log_lik += float(np.sum(nz * np.log(nz))) - s * math.log(s)

    n_params = (n_categories - 1) * n_parent_configs
    penalty = -0.5 * n_params * math.log(n_samples)
    return log_lik + penalty


# ===========================================================================
# Section 5b – Phase 2 directed fairness penalty (paper Eq. 7-8)
# ===========================================================================

def _compute_markov_blanket(
    graph_matrix: np.ndarray,
    y_idx: int,
) -> Set[int]:
    """
    Compute the Markov Blanket of Y in a fully oriented DAG.

    MB(Y) = Pa(Y) ∪ Ch(Y) ∪ Pa(Ch(Y)) \ {Y}

    Uses the GES_fair graph encoding:
      graph_matrix[i, j] == TAIL and graph_matrix[j, i] == ARROW  ⇒  i → j
    """
    n = graph_matrix.shape[0]
    # Parents of Y: nodes i such that i → Y
    parents_y: Set[int] = {
        i for i in range(n)
        if i != y_idx
        and graph_matrix[i, y_idx] == _TAIL
        and graph_matrix[y_idx, i] == _ARROW
    }
    # Children of Y: nodes j such that Y → j
    children_y: Set[int] = {
        j for j in range(n)
        if j != y_idx
        and graph_matrix[y_idx, j] == _TAIL
        and graph_matrix[j, y_idx] == _ARROW
    }
    # Co-parents: other parents of children of Y (excluding Y itself)
    co_parents: Set[int] = set()
    for ch in children_y:
        for i in range(n):
            if i != y_idx and i != ch:
                if graph_matrix[i, ch] == _TAIL and graph_matrix[ch, i] == _ARROW:
                    co_parents.add(i)
    return parents_y | children_y | co_parents


def compute_phase2_penalty(
    dag_graph_matrix: np.ndarray,
    data: np.ndarray,
    proxy_load: Dict[int, float],
    outcome_index: int,
    beta: float = 0.0,
) -> float:
    """
    Compute the total directed fairness penalty Ψ_Total(G) for a fully oriented
    DAG (Phase 2, Eq. 7-8 in the paper).

      Ψ_Total(G) = Σ_{V_i ∈ V} ρ_G(V_i) · Σ_{V_j ∈ Pa_G(V_i)} φ*(V_j)

    where the graph-aware outcome relevance ρ_G(V_i) is:

      ρ_G(V_i) = 1                    if V_i == Y or V_i ∈ MB_G(Y)
               = β · NMI(V_i, Y)      otherwise

    β ∈ [0, 1] controls the penalty scope beyond MB(Y).  When β = 0 (default),
    only edges within the Markov Blanket of Y are penalised — the tractable
    special case from Section 4.3.5 of the paper.

    Parameters
    ----------
    dag_graph_matrix : G.graph from a fully oriented DAG (N_vars × N_vars int array)
    data             : original data array (N_samples × N_vars)
    proxy_load       : precomputed dict from compute_proxy_load()
    outcome_index    : column index of the outcome variable Y
    beta             : discount for nodes outside MB(Y), ∈ [0, 1].  Default 0.0.

    Returns
    -------
    float : Ψ_Total ≥ 0
    """
    n_vars = dag_graph_matrix.shape[0]

    # Compute MB(Y) from the DAG structure
    mb_y = _compute_markov_blanket(dag_graph_matrix, outcome_index)

    # Precompute H(Y) only when needed (β > 0)
    hy = discrete_entropy(data[:, outcome_index]) if beta > 0 else 0.0

    total = 0.0
    for i in range(n_vars):
        # Determine ρ_G(V_i)
        if i == outcome_index or i in mb_y:
            rho = 1.0
        elif beta > 0:
            hi = discrete_entropy(data[:, i])
            denom = min(hi, hy)
            if denom < 1e-12:
                rho = 0.0
            else:
                mi = discrete_mutual_information(data[:, i], data[:, outcome_index])
                rho = beta * min(1.0, mi / denom)
        else:
            rho = 0.0  # β == 0: only MB(Y) nodes contribute

        if rho == 0.0:
            continue

        # Sum φ*(V_j) over parents of V_i in the DAG
        parent_proxy_sum = sum(
            proxy_load.get(j, 0.0)
            for j in range(n_vars)
            if j != i
            and dag_graph_matrix[j, i] == _TAIL
            and dag_graph_matrix[i, j] == _ARROW
        )
        total += rho * parent_proxy_sum

    return total
