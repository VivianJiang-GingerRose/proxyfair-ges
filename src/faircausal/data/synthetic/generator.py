from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.faircausal.core.fairness_scoring import compute_proxy_strength_matrix, cramers_v
from src.faircausal.data.synthetic.archetype_definitions import ArchetypeSpec, get_archetype, observed_mask
from src.faircausal.data.synthetic.calibration import (
    CalibrationResult,
    calibrate_sv_analytical,
    calibrate_sv_coefficient,
    discretize_proxy_against_sensitive,
)
from src.faircausal.data.synthetic.discretize import (
    _discretize_column_quantile,
    check_category_diagnostics,
    discretize_matrix,
)
from src.faircausal.data.synthetic.sem import compute_true_pse, generate_continuous


@dataclass(frozen=True)
class SyntheticDataset:
    data: np.ndarray
    variable_names: List[str]
    protected_indices: List[int]
    outcome_index: int
    proxy_index: int
    legit_index: int
    true_edges: List[tuple[str, str]]
    true_coefficients: Dict[tuple[str, str], float]
    true_pse: Dict[str, float]
    archetype_id: int
    calibration: Dict[str, Any]
    diagnostics: Dict[str, Any]
    proxy_indices: Optional[List[int]] = None
    legit_indices: Optional[List[int]] = None
    displacement_paths: Optional[List[Tuple[str, str, str]]] = None
    continuous_data: Optional[np.ndarray] = None



def _project_observed(archetype: ArchetypeSpec, full_data: np.ndarray) -> tuple[np.ndarray, List[str]]:
    mask = observed_mask(archetype)
    obs_names = [v for v, m in zip(archetype.all_variables, mask) if m]
    return full_data[:, mask], obs_names


def _simple_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    if float(np.var(x)) <= 1e-12 or float(np.var(y)) <= 1e-12:
        return float("nan")
    corr = float(np.corrcoef(x, y)[0, 1])
    return float(corr**2)


def _displacement_calibration(
    observed_continuous: np.ndarray,
    data_disc: np.ndarray,
    variable_names: List[str],
    n_bins: int,
) -> Dict[str, Any]:
    idx = {name: i for i, name in enumerate(variable_names)}
    s1 = idx["S1"]
    s2 = idx["S2"]
    v1 = idx["V1"]
    v2 = idx["V2"]

    phi_strength = compute_proxy_strength_matrix(data_disc, variable_names, [s1, s2])
    corr_s1_s2 = float(np.corrcoef(observed_continuous[:, s1], observed_continuous[:, s2])[0, 1])
    r2_v1_s1 = _simple_r2(observed_continuous[:, v1], observed_continuous[:, s1])
    r2_v2_s2 = _simple_r2(observed_continuous[:, v2], observed_continuous[:, s2])

    return {
        "phi_target": 0.5,
        "phi_star_target": 0.5,
        "protected_corr_target": 0.5,
        "proxy_r2_target": 0.5,
        "phi_achieved": float(np.nanmean([r2_v1_s1, r2_v2_s2])),
        "phi_r2_achieved": float(np.nanmean([r2_v1_s1, r2_v2_s2])),
        "phi_nmi_achieved": float(np.nanmean([phi_strength.get((v1, s1), np.nan), phi_strength.get((v2, s2), np.nan)])),
        "phi_metric": "r2_continuous",
        "n_bisection_steps": 0,
        "n_bins": int(n_bins),
        "corr_s1_s2_continuous": corr_s1_s2,
        "r2_v1_s1_continuous": r2_v1_s1,
        "r2_v2_s2_continuous": r2_v2_s2,
        "nmi_v1_s1_discrete": float(phi_strength.get((v1, s1), np.nan)),
        "nmi_v2_s2_discrete": float(phi_strength.get((v2, s2), np.nan)),
        "cramers_v_s1_s2_discrete": float(cramers_v(data_disc[:, s1], data_disc[:, s2])),
        "cramers_v_v1_s1_discrete": float(cramers_v(data_disc[:, v1], data_disc[:, s1])),
        "cramers_v_v2_s2_discrete": float(cramers_v(data_disc[:, v2], data_disc[:, s2])),
    }


def generate_synthetic_dataset(
    archetype_id: int,
    N: int,
    phi_target: float,
    n_bins: int = 3,
    seed: Optional[int] = None,
    calibration_tol: float = 0.05,
    confounding_strength: Optional[float] = None,
) -> Dict[str, Any]:
    archetype = get_archetype(archetype_id)

    calibration_result = CalibrationResult(coefficient=0.0, phi_achieved=0.0, n_steps=0)
    sv_coeff: Optional[float] = None

    if ("S", "V") in archetype.edges:
        if archetype.archetype_id == 1:
            rng = np.random.default_rng(seed)
            sv_coeff_mag = calibrate_sv_analytical(phi_target=float(phi_target))
            sv_sign = rng.choice([-1.0, 1.0])
            sv_coeff = float(sv_sign * sv_coeff_mag)
            calibration_result = CalibrationResult(coefficient=sv_coeff, phi_achieved=float(phi_target), n_steps=0)
        else:
            calibration_result = calibrate_sv_coefficient(
                phi_target=phi_target,
                n_bins=n_bins,
                N=max(N, 3000),
                extra_noise_var=0.0,
                tol=calibration_tol,
                seed=seed,
                return_details=True,
            )
            sv_coeff = calibration_result.coefficient

    sem_out = generate_continuous(
        archetype=archetype,
        N=N,
        sv_coefficient=sv_coeff,
        seed=seed,
        confounding_strength=confounding_strength,
    )

    observed_continuous, variable_names = _project_observed(archetype, sem_out.full_data)
    data_disc = discretize_matrix(observed_continuous, n_bins=n_bins, method="quantile")

    protected_indices = [variable_names.index(v) for v in archetype.protected_variables if v in variable_names]
    if not protected_indices:
        raise ValueError("No protected variable present in observed data")

    outcome_index = variable_names.index(archetype.outcome_variable)
    proxy_index = variable_names.index(archetype.proxy_variable)
    legit_index = variable_names.index(archetype.legit_variable)
    proxy_indices = [variable_names.index(v) for v in archetype.proxy_variables if v in variable_names]
    if not proxy_indices:
        proxy_indices = [proxy_index]
    legit_indices = [variable_names.index(v) for v in archetype.legit_variables if v in variable_names]
    if not legit_indices:
        legit_indices = [legit_index]

    if archetype.archetype_id != 4:
        # Keep Y binary for downstream classification/fairness evaluation.
        data_disc[:, outcome_index] = _discretize_column_quantile(
            observed_continuous[:, outcome_index],
            n_bins=2,
        )

        # Keep proxy discretization consistent with calibration step.
        sensitive_idx = protected_indices[0]
        proxy_disc = discretize_proxy_against_sensitive(
            observed_continuous[:, proxy_index],
            observed_continuous[:, sensitive_idx],
        )
        data_disc[:, proxy_index] = proxy_disc.astype(np.int64)

    diagnostics = {
        "N": int(N),
        "seed": int(seed) if seed is not None else None,
        "category": check_category_diagnostics(data_disc, variable_names),
    }

    if archetype.archetype_id == 4:
        calibration = _displacement_calibration(
            observed_continuous=observed_continuous,
            data_disc=data_disc,
            variable_names=variable_names,
            n_bins=n_bins,
        )
    else:
        sensitive_idx = protected_indices[0]
        phi_nmi_achieved = float("nan")
        phi_cramers_v_achieved = float("nan")
        try:
            phi_strength = compute_proxy_strength_matrix(data_disc, variable_names, [protected_indices[0]])
            phi_nmi_achieved = float(phi_strength[(proxy_index, protected_indices[0])])
        except Exception:
            pass
        try:
            phi_cramers_v_achieved = float(cramers_v(data_disc[:, proxy_index], data_disc[:, sensitive_idx]))
        except Exception:
            pass

        v_cont = observed_continuous[:, proxy_index].astype(np.float64)
        s_cont = observed_continuous[:, sensitive_idx].astype(np.float64)
        phi_r2_achieved = _simple_r2(v_cont, s_cont)

        calibration = {
            "phi_target": float(phi_target),
            "phi_star_target": float(phi_target),
            "phi_achieved": float(phi_r2_achieved),
            "phi_r2_achieved": float(phi_r2_achieved),
            "phi_nmi_achieved": float(phi_nmi_achieved),
            "phi_star_cramers_v_achieved": float(phi_cramers_v_achieved),
            "phi_star_cramers_v_abs_error": float(abs(phi_cramers_v_achieved - float(phi_target))),
            "phi_metric": "r2_continuous",
            "n_bisection_steps": int(calibration_result.n_steps),
            "n_bins": int(n_bins),
            "sv_coefficient": float(calibration_result.coefficient),
        }

    true_edges = [(src, dst) for src, dst in archetype.edges if src in variable_names and dst in variable_names]

    true_pse = compute_true_pse(archetype, sem_out.coefficients)

    out = SyntheticDataset(
        data=data_disc.astype(np.int64),
        variable_names=variable_names,
        protected_indices=protected_indices,
        outcome_index=outcome_index,
        proxy_index=proxy_index,
        legit_index=legit_index,
        true_edges=true_edges,
        true_coefficients=sem_out.coefficients,
        true_pse=true_pse,
        archetype_id=archetype.archetype_id,
        calibration=calibration,
        diagnostics=diagnostics,
        proxy_indices=proxy_indices,
        legit_indices=legit_indices,
        displacement_paths=[("S1", "V1", "Y"), ("S2", "V2", "Y")] if archetype.archetype_id == 4 else None,
        continuous_data=observed_continuous.copy(),
    )

    result = {
        "data": out.data,
        "variable_names": out.variable_names,
        "protected_indices": out.protected_indices,
        "outcome_index": out.outcome_index,
        "proxy_index": out.proxy_index,
        "legit_index": out.legit_index,
        "proxy_indices": out.proxy_indices,
        "legit_indices": out.legit_indices,
        "true_edges": out.true_edges,
        "true_coefficients": out.true_coefficients,
        "true_pse": out.true_pse,
        "archetype_id": out.archetype_id,
        "calibration": out.calibration,
        "diagnostics": out.diagnostics,
        "continuous_data": out.continuous_data,
    }
    if out.displacement_paths is not None:
        result["displacement_paths"] = out.displacement_paths
    return result
