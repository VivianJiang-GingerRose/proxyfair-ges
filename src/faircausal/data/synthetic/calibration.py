from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.faircausal.core.fairness_scoring import discrete_entropy, discrete_mutual_information
from src.faircausal.data.synthetic.discretize import discretize_matrix


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibrationResult:
    coefficient: float
    phi_achieved: float
    n_steps: int


def _nmi(x: np.ndarray, s: np.ndarray) -> float:
    hx = discrete_entropy(x)
    hs = discrete_entropy(s)
    denom = min(hx, hs)
    if denom <= 1e-12:
        return 0.0
    mi = discrete_mutual_information(x, s)
    return float(min(1.0, mi / (denom + 1e-12)))


def discretize_proxy_against_sensitive(
    v: np.ndarray,
    s: np.ndarray,
    n_candidates: int = 41,
) -> np.ndarray:
    """Binary-discretize proxy by choosing threshold that maximizes NMI with S."""
    v = np.asarray(v, dtype=np.float64).ravel()
    s = np.asarray(s).ravel().astype(np.int64)

    if v.size == 0:
        return np.array([], dtype=np.int64)

    percentiles = np.linspace(1.0, 99.0, num=max(5, n_candidates))
    thresholds = np.unique(np.percentile(v, percentiles))
    thresholds = np.unique(np.concatenate([thresholds, np.array([np.median(v)])]))

    best_phi = -1.0
    best_disc = (v > float(np.median(v))).astype(np.int64)

    for t in thresholds:
        disc = (v > float(t)).astype(np.int64)
        if disc.min() == disc.max():
            continue
        phi = _nmi(disc, s)
        if phi > best_phi:
            best_phi = phi
            best_disc = disc

    return best_disc


def _simulate_phi(
    b: float,
    s_samples: np.ndarray,
    eps_samples: np.ndarray,
    n_bins: int,
) -> float:
    v = b * s_samples + eps_samples
    v_disc = discretize_proxy_against_sensitive(v, s_samples)
    return _nmi(v_disc, s_samples)


def calibrate_sv_coefficient(
    phi_target: float,
    n_bins: int,
    N: int,
    noise_var: float = 1.0,
    extra_noise_var: float = 0.0,
    tol: float = 0.05,
    max_iter: int = 50,
    seed: Optional[int] = None,
    b_lo: float = 0.01,
    b_hi: float = 10.0,
    return_details: bool = False,
):
    if not (0.0 < phi_target < 1.0):
        raise ValueError("phi_target must be in (0, 1)")
    if N <= 10:
        raise ValueError("N must be > 10 for stable calibration")

    rng = np.random.default_rng(seed)
    s_samples = rng.binomial(1, 0.5, size=N).astype(np.int64)
    total_noise_var = float(noise_var) + float(extra_noise_var)
    if total_noise_var <= 0.0:
        raise ValueError("noise_var + extra_noise_var must be positive")
    eps_samples = rng.normal(loc=0.0, scale=np.sqrt(total_noise_var), size=N)

    phi_lo = _simulate_phi(b_lo, s_samples, eps_samples, n_bins)
    phi_hi = _simulate_phi(b_hi, s_samples, eps_samples, n_bins)

    if phi_target < phi_lo - tol or phi_target > phi_hi + tol:
        raise CalibrationError(
            f"Target phi={phi_target:.4f} out of reachable range [{phi_lo:.4f}, {phi_hi:.4f}] for N={N}, n_bins={n_bins}"
        )

    best_b = b_lo
    best_phi = phi_lo
    for step in range(1, max_iter + 1):
        b_mid = 0.5 * (b_lo + b_hi)
        phi_mid = _simulate_phi(b_mid, s_samples, eps_samples, n_bins)

        if abs(phi_mid - phi_target) < abs(best_phi - phi_target):
            best_b = b_mid
            best_phi = phi_mid

        if abs(phi_mid - phi_target) <= tol:
            result = CalibrationResult(coefficient=float(b_mid), phi_achieved=float(phi_mid), n_steps=step)
            return result if return_details else result.coefficient

        if phi_mid < phi_target:
            b_lo = b_mid
        else:
            b_hi = b_mid

    if abs(best_phi - phi_target) > tol:
        raise CalibrationError(
            f"Calibration failed after {max_iter} steps. target={phi_target:.4f}, achieved={best_phi:.4f}"
        )

    result = CalibrationResult(coefficient=float(best_b), phi_achieved=float(best_phi), n_steps=max_iter)
    return result if return_details else result.coefficient


def calibrate_sv_analytical(phi_target: float) -> float:
    # For binary S in {0,1} with unit Gaussian noise in V = b_SV * S + eps,
    # target proxy strength phi* is achieved by b_SV = 2 * sqrt(phi* / (1 - phi*)).
    if not (0.0 < phi_target < 1.0):
        raise ValueError("phi_target must be in (0, 1)")
    numerator = float(phi_target)
    denominator = 0.25 * (1.0 - float(phi_target))
    if denominator <= 0.0:
        raise ValueError("Invalid denominator when calibrating sv coefficient")
    beta_sv_sq = numerator / denominator
    return float(np.sqrt(max(beta_sv_sq, 0.0)))


def proxy_strength_monotone_check(
    coefficients: Tuple[float, ...],
    n_bins: int,
    N: int,
    seed: Optional[int] = None,
) -> Dict[float, float]:
    rng = np.random.default_rng(seed)
    s_samples = rng.binomial(1, 0.5, size=N).astype(np.int64)
    eps_samples = rng.normal(loc=0.0, scale=1.0, size=N)

    phis: Dict[float, float] = {}
    for b in coefficients:
        phis[float(b)] = _simulate_phi(float(abs(b)), s_samples, eps_samples, n_bins)
    return phis
