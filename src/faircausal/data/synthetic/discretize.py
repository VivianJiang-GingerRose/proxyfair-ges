from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _discretize_column_quantile(col: np.ndarray, n_bins: int) -> np.ndarray:
    series = pd.Series(col)
    # duplicates='drop' avoids errors when quantile boundaries collapse
    binned = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    if binned.isna().all():
        return np.zeros_like(col, dtype=np.int64)
    if binned.isna().any():
        fill_val = int(np.nanmedian(binned.to_numpy())) if not np.isnan(np.nanmedian(binned.to_numpy())) else 0
        binned = binned.fillna(fill_val)
    return binned.astype(np.int64).to_numpy()


def discretize_matrix(
    continuous_data: np.ndarray,
    n_bins: int = 3,
    method: str = "quantile",
    preserve_binary: bool = True,
) -> np.ndarray:
    if method != "quantile":
        raise ValueError("Only method='quantile' is currently supported")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    data = np.asarray(continuous_data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("continuous_data must be a 2D array")

    out = np.zeros_like(data, dtype=np.int64)
    for j in range(data.shape[1]):
        col = data[:, j]
        unique_vals = np.unique(col)
        if preserve_binary and len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
            out[:, j] = col.astype(np.int64)
        else:
            out[:, j] = _discretize_column_quantile(col, n_bins=n_bins)
    return out


def check_category_diagnostics(discrete_data: np.ndarray, variable_names: List[str]) -> Dict[str, Dict[str, float]]:
    data = np.asarray(discrete_data)
    if data.ndim != 2:
        raise ValueError("discrete_data must be a 2D array")

    diagnostics: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(variable_names):
        vals, counts = np.unique(data[:, j], return_counts=True)
        min_count = int(np.min(counts)) if counts.size else 0
        diagnostics[name] = {
            "n_categories": float(len(vals)),
            "min_category_freq": float(min_count),
            "chi_square_safe": float(min_count >= 5),
        }
    return diagnostics
