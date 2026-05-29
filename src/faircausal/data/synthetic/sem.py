from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.faircausal.data.synthetic.archetype_definitions import ArchetypeSpec


@dataclass(frozen=True)
class ContinuousSEMResult:
    all_variable_names: Tuple[str, ...]
    full_data: np.ndarray
    coefficients: Dict[Tuple[str, str], float]


def topological_order(edges: List[Tuple[str, str]], variables: List[str]) -> List[str]:
    indeg = {v: 0 for v in variables}
    children: Dict[str, List[str]] = {v: [] for v in variables}
    for src, dst in edges:
        if src not in indeg or dst not in indeg:
            raise ValueError(f"Unknown variable in edge ({src}, {dst})")
        indeg[dst] += 1
        children[src].append(dst)

    queue = [v for v in variables if indeg[v] == 0]
    order: List[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for c in children[node]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)

    if len(order) != len(variables):
        raise ValueError("Graph contains a cycle; topological order not possible")
    return order


def _sample_signed_uniform(rng: np.random.Generator, low: float = 0.5, high: float = 1.5) -> float:
    mag = rng.uniform(low, high)
    sign = rng.choice([-1.0, 1.0])
    return float(sign * mag)


def _generate_displacement_continuous(
    archetype: ArchetypeSpec,
    N: int,
    seed: Optional[int],
) -> ContinuousSEMResult:
    rng = np.random.default_rng(seed)
    variables = list(archetype.all_variables)
    idx = {v: i for i, v in enumerate(variables)}
    data = np.zeros((N, len(variables)), dtype=np.float64)

    corr_target = 0.5
    proxy_r2_target = 0.5
    a = float(np.sqrt(corr_target / (1.0 - corr_target)))
    var_s = a**2 + 1.0
    b = float(np.sqrt(proxy_r2_target / ((1.0 - proxy_r2_target) * var_s)))
    c1 = 1.0
    c2 = 1.0
    c3 = 1.5

    data[:, idx["C"]] = rng.normal(0.0, 1.0, size=N)
    data[:, idx["S1"]] = a * data[:, idx["C"]] + rng.normal(0.0, 1.0, size=N)
    data[:, idx["S2"]] = a * data[:, idx["C"]] + rng.normal(0.0, 1.0, size=N)
    data[:, idx["V1"]] = b * data[:, idx["S1"]] + rng.normal(0.0, 1.0, size=N)
    data[:, idx["V2"]] = b * data[:, idx["S2"]] + rng.normal(0.0, 1.0, size=N)
    data[:, idx["Z"]] = rng.normal(0.0, 1.0, size=N)
    data[:, idx["Y"]] = (
        c1 * data[:, idx["V1"]]
        + c2 * data[:, idx["V2"]]
        + c3 * data[:, idx["Z"]]
        + rng.normal(0.0, 1.0, size=N)
    )

    coeffs: Dict[Tuple[str, str], float] = {
        ("C", "S1"): a,
        ("C", "S2"): a,
        ("S1", "V1"): b,
        ("S2", "V2"): b,
        ("V1", "Y"): c1,
        ("V2", "Y"): c2,
        ("Z", "Y"): c3,
    }
    return ContinuousSEMResult(
        all_variable_names=tuple(variables),
        full_data=data,
        coefficients=coeffs,
    )


def generate_continuous(
    archetype: ArchetypeSpec,
    N: int,
    sv_coefficient: Optional[float],
    seed: Optional[int] = None,
    confounding_strength: Optional[float] = None,
) -> ContinuousSEMResult:
    if N <= 0:
        raise ValueError("N must be positive")

    if archetype.archetype_id == 4:
        return _generate_displacement_continuous(archetype=archetype, N=N, seed=seed)

    rng = np.random.default_rng(seed)
    variables = list(archetype.all_variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}

    parents: Dict[str, List[str]] = {v: [] for v in variables}
    coeffs: Dict[Tuple[str, str], float] = {}

    for src, dst in archetype.edges:
        parents[dst].append(src)
        if (src, dst) == ("S", "V") and sv_coefficient is not None:
            coeffs[(src, dst)] = float(sv_coefficient)
        elif archetype.archetype_id == 1 and (src, dst) == ("V", "Y"):
            # Store the S-mediated Y coefficient in (V, Y) to preserve true PSE accounting.
            coeffs[(src, dst)] = _sample_signed_uniform(rng)
        elif archetype.archetype_id == 3 and src == "U" and dst in {"S", "V"} and confounding_strength is not None:
            coeffs[(src, dst)] = float(abs(confounding_strength))
        else:
            coeffs[(src, dst)] = _sample_signed_uniform(rng)

    order = topological_order(list(archetype.edges), variables)
    data = np.zeros((N, len(variables)), dtype=np.float64)

    protected_set = set(archetype.protected_variables)

    for v in order:
        idx = var_to_idx[v]
        v_parents = parents[v]

        if v in protected_set and len(v_parents) == 0:
            data[:, idx] = rng.binomial(1, 0.5, size=N).astype(np.float64)
            continue

        linear = np.zeros(N, dtype=np.float64)
        for p in v_parents:
            p_idx = var_to_idx[p]
            linear += coeffs[(p, v)] * data[:, p_idx]

        noise = rng.normal(0.0, 1.0, size=N)

        if v in protected_set:
            # Parent-influenced protected variables are thresholded to remain binary.
            logits = linear + noise
            threshold = np.median(logits)
            data[:, idx] = (logits > threshold).astype(np.float64)
        else:
            data[:, idx] = linear + noise

    return ContinuousSEMResult(
        all_variable_names=tuple(variables),
        full_data=data,
        coefficients=coeffs,
    )


def compute_true_pse(archetype: ArchetypeSpec, coefficients: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    if archetype.archetype_id == 4:
        y = archetype.outcome_variable
        pse_s1 = float(coefficients.get(("S1", "V1"), 0.0) * coefficients.get(("V1", y), 0.0))
        pse_s2 = float(coefficients.get(("S2", "V2"), 0.0) * coefficients.get(("V2", y), 0.0))
        return {
            "total": float(pse_s1 + pse_s2),
            "direct": 0.0,
            "indirect_proxy": float(pse_s1 + pse_s2),
            "pse_s1": pse_s1,
            "pse_s2": pse_s2,
            "pse_total": float(pse_s1 + pse_s2),
        }

    s = archetype.protected_variables[0]
    y = archetype.outcome_variable
    v = archetype.proxy_variable

    direct = float(coefficients.get((s, y), 0.0))
    indirect_proxy = 0.0

    if (s, v) in coefficients and (v, y) in coefficients:
        indirect_proxy = float(coefficients[(s, v)] * coefficients[(v, y)])

    total = direct + indirect_proxy
    return {
        "total": float(total),
        "direct": float(direct),
        "indirect_proxy": float(indirect_proxy),
    }
