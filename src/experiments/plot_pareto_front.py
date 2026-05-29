"""Plot scatter-only Pareto fronts for fairness-utility trade-offs.

Scans results directories, extracts TE/PSE/AUROC metrics, and saves 4 PDFs:
- Soft-only: AUROC vs |TE|
- Soft-only: AUROC vs |PSE|
- Soft+hard: AUROC vs |TE|
- Soft+hard: AUROC vs |PSE|

Usage::

    poetry run python src/experiments/plot_pareto_front.py --dataset compas
    poetry run python src/experiments/plot_pareto_front.py \\
        --dataset compas \\
        --output results/paper_results
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Run-dir prefix → series key
SERIES_SOFT_ONLY = "soft_only"
SERIES_SOFT_HARD = "soft_hard"

PREFIX_SOFT_ONLY = "soft_fairness_only"
PREFIX_SOFT_HARD_ALT = "soft_fairness_"   # catches both patterns (checked after ONLY)
PREFIX_BASELINE = "baseline"
PREFIX_HARD_ONLY = "fairness"

DISPLAY_NAMES = {
    SERIES_SOFT_ONLY: "Soft Constraint Only",
    SERIES_SOFT_HARD: "Soft + Hard Constraints",
}


# ---------------------------------------------------------------------------
# Low-level helpers (same semantics as compile_causal_fairness_analysis_table)
# ---------------------------------------------------------------------------

def _parse_value_with_pm(cell: object) -> float:
    text = str(cell)
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _maybe_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _mean_finite(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else None


# ---------------------------------------------------------------------------
# Scanning a single run directory
# ---------------------------------------------------------------------------

def _classify_run_dir(dir_name: str) -> Optional[Tuple[str, bool]]:
    """Return (series_key, is_anchor) or None if not a recognised run dir."""
    if dir_name.startswith(PREFIX_SOFT_ONLY):
        return SERIES_SOFT_ONLY, False
    # soft_fairness_only handled above; now check soft_fairness lambda runs
    # New naming: soft_fairness_lambda_... or old naming: soft_fairness_YYYYMMDD_HHMMSS_lambda_...
    if re.match(r"soft_fairness(?:_lambda_|_\d{8}_)", dir_name):
        return SERIES_SOFT_HARD, False
    if dir_name.startswith(PREFIX_BASELINE):
        return SERIES_SOFT_ONLY, True
    if dir_name.startswith(PREFIX_HARD_ONLY):
        return SERIES_SOFT_HARD, True
    return None


def _find_results_in_dir(run_dir: Path) -> Optional[Tuple[str, Path, Path, Path]]:
    """Find (exp_stem, config_json, cf_csv, ml_csv) under run_dir/results/."""
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return None
    configs = list(results_dir.glob("*_config.json"))
    if not configs:
        return None
    # If multiple configs exist, prefer the most specific (longest stem)
    config_path = max(configs, key=lambda p: len(p.stem))
    exp_stem = config_path.stem.replace("_config", "")
    cf_path = results_dir / f"{exp_stem}_cf_results.csv"
    ml_path = results_dir / f"{exp_stem}_ml_results.csv"
    if not cf_path.exists() or not ml_path.exists():
        return None
    return exp_stem, config_path, cf_path, ml_path


def _read_lambda_from_config(config_path: Path) -> Optional[float]:
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return float(cfg["dataset_config"]["soft_fairness_settings"]["lambda"])
    except Exception:
        return None


def _extract_metrics(
    cf_path: Path,
    ml_path: Path,
    protected_attrs: List[str],
) -> Optional[Dict]:
    """Extract TE, PSE, AUROC metrics from one run's CSV files."""
    try:
        cf_df = pd.read_csv(cf_path)
        ml_df = pd.read_csv(ml_path)
    except Exception:
        return None

    standard = cf_df[cf_df["data_type"] == "standard"].copy()
    if standard.empty:
        return None

    dag_ids = sorted(int(d) for d in standard["dag_id"].dropna().unique())

    # --- Per-attribute TE and PSE, averaged over all DAGs ---
    te_per_attr: Dict[str, List[float]] = {a: [] for a in protected_attrs}
    pse_per_attr: Dict[str, List[float]] = {a: [] for a in protected_attrs}

    for dag_id in dag_ids:
        dag_rows = standard[standard["dag_id"] == dag_id]
        for attr in protected_attrs:
            single_rows = dag_rows[
                (dag_rows["group_name"] == f"single_{attr}")
                & (dag_rows["attribute_name"] == attr)
                & (~dag_rows["attribute_has_error"].astype(bool))
            ]
            if single_rows.empty:
                continue
            row = single_rows.iloc[0]

            te_val = _maybe_float(row.get(f"group_{attr}_total_effect"))
            if te_val is not None:
                te_per_attr[attr].append(abs(te_val))

            # PSE: try column then fall back to |TE - NDE|
            pse_cols = [
                c for c in row.index
                if c.startswith(f"group_{attr}_path_") and c.endswith("_pse_score")
            ]
            pse_vals = [_maybe_float(row.get(c)) for c in pse_cols]
            pse_finite = [abs(v) for v in pse_vals if v is not None]
            if pse_finite:
                pse_per_attr[attr].append(float(sum(pse_finite) / len(pse_finite)))
            else:
                nde_val = _maybe_float(row.get(f"group_{attr}_nde_score"))
                if te_val is not None and nde_val is not None:
                    pse_per_attr[attr].append(abs(te_val - nde_val))

    # Average absolute TE / PSE across all attributes
    all_te = [v for vals in te_per_attr.values() for v in vals]
    all_pse = [v for vals in pse_per_attr.values() for v in vals]
    te = _mean_finite(all_te)
    pse = _mean_finite(all_pse)

    # Also grab the overall_te_mean (aggregate from first per-DAG row) for reference
    overall_te_vals = []
    for dag_id in dag_ids:
        dag_rows = standard[standard["dag_id"] == dag_id]
        v = _maybe_float(dag_rows.iloc[0].get("overall_te_mean"))
        if v is not None:
            overall_te_vals.append(abs(v))
    overall_te = _mean_finite(overall_te_vals)

    # --- AUROC from xgboost ---
    xgb = ml_df[
        (ml_df["data_type"] == "standard") & (ml_df["model_type"] == "xgboost")
    ]
    if xgb.empty:
        xgb = ml_df[ml_df["model_type"] == "xgboost"]

    def _mean_col(frame: pd.DataFrame, col: str) -> float:
        if col not in frame.columns:
            return float("nan")
        vals = [_parse_value_with_pm(v) for v in frame[col].tolist() if not pd.isna(v)]
        finite = [v for v in vals if not math.isnan(v)]
        return float(sum(finite) / len(finite)) if finite else float("nan")

    auroc = _mean_col(xgb, "AUROC")
    f1 = _mean_col(xgb, "f1")

    return {
        "te": te if te is not None else float("nan"),
        "pse": pse if pse is not None else float("nan"),
        "overall_te": overall_te if overall_te is not None else float("nan"),
        "auroc": auroc,
        "f1": f1,
        "n_dags": len(dag_ids),
    }


# ---------------------------------------------------------------------------
# Scan all runs for a dataset
# ---------------------------------------------------------------------------

def scan_dataset(
    dataset_root: Path,
    protected_attrs: List[str],
) -> pd.DataFrame:
    """Return a DataFrame with one row per (series, lambda) data point."""
    records = []

    for run_dir in sorted(dataset_root.iterdir()):
        if not run_dir.is_dir():
            continue

        classification = _classify_run_dir(run_dir.name)
        if classification is None:
            continue
        series_key, is_anchor = classification

        found = _find_results_in_dir(run_dir)
        if found is None:
            continue
        exp_stem, config_path, cf_path, ml_path = found

        # Lambda
        if is_anchor:
            lam = 0.0
        else:
            lam = _read_lambda_from_config(config_path)
            if lam is None:
                print(f"  [skip] could not read lambda from {config_path}")
                continue

        metrics = _extract_metrics(cf_path, ml_path, protected_attrs)
        if metrics is None:
            print(f"  [skip] failed to extract metrics from {run_dir.name}")
            continue

        records.append({
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "series": series_key,
            "is_anchor": is_anchor,
            "lambda": lam,
            **metrics,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # De-duplicate: for the same (series, lambda, is_anchor), keep the latest run
    # (sort by run_name which contains the timestamp)
    df = (
        df.sort_values("run_name")
        .drop_duplicates(subset=["series", "lambda", "is_anchor"], keep="last")
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------------
# Pareto front computation
# ---------------------------------------------------------------------------

def pareto_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_higher: bool = True,
    y_higher: bool = False,
) -> np.ndarray:
    """Boolean mask: True for Pareto-optimal (non-dominated) points."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Does j dominate i?
            def _weakly_better(a: float, b: float, higher: bool) -> bool:
                return (a >= b) if higher else (a <= b)

            def _strictly_better(a: float, b: float, higher: bool) -> bool:
                return (a > b) if higher else (a < b)

            x_weak = _weakly_better(x[j], x[i], x_higher)
            y_weak = _weakly_better(y[j], y[i], y_higher)
            x_strict = _strictly_better(x[j], x[i], x_higher)
            y_strict = _strictly_better(y[j], y[i], y_higher)

            if x_weak and y_weak and (x_strict or y_strict):
                dominated[i] = True
                break

    return ~dominated


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_pareto_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    y_metric: str,
    y_label: str,
    cmap: str = "plasma",
    anchor_color: str = "#d62728",
    series_title: str = "",
    annotate_lambdas: bool = True,
    annotate_pareto_lambdas: bool = True,
    color_norm: Optional[mcolors.Normalize] = None,
) -> Optional[object]:
    """Draw scatter + Pareto frontier + anchor on one axes panel."""
    sweep = df[~df["is_anchor"]].dropna(subset=["auroc", y_metric]).copy()
    anchors = df[df["is_anchor"]].dropna(subset=["auroc", y_metric]).copy()

    if sweep.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return None

    # Filter out NaN auroc/y_metric
    sweep = sweep[np.isfinite(sweep["auroc"]) & np.isfinite(sweep[y_metric])]
    if sweep.empty:
        return None

    lambdas = sweep["lambda"].values
    log_lam = np.log10(lambdas)
    norm = color_norm if color_norm is not None else mcolors.Normalize(vmin=log_lam.min(), vmax=log_lam.max())
    cmap_obj = matplotlib.colormaps[cmap]

    sc = ax.scatter(
        sweep["auroc"],
        sweep[y_metric],
        c=log_lam,
        cmap=cmap_obj,
        norm=norm,
        s=112,
        zorder=3,
        alpha=0.95,
        edgecolors="white",
        linewidths=0.4,
        label=r"$\lambda_f$ sweep",
    )

    if annotate_lambdas:
        for _, row in sweep.iterrows():
            lam_str = f"{row['lambda']:g}"
            ax.annotate(
                f"$\\lambda_f={lam_str}$",
                (row["auroc"], row[y_metric]),
                fontsize=8,
                ha="left",
                va="bottom",
                xytext=(2, 2),
                textcoords="offset points",
                zorder=5,
                alpha=0.78,
            )

    # Pareto frontier
    x_arr = sweep["auroc"].values
    y_arr = sweep[y_metric].values
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() >= 2:
        sweep_valid = sweep.loc[valid].copy()
        pm = pareto_mask(
            sweep_valid["auroc"].values,
            sweep_valid[y_metric].values,
            x_higher=True,
            y_higher=False,
        )
        pareto_df = sweep_valid.loc[pm].copy()
        px, py = pareto_df["auroc"].values, pareto_df[y_metric].values
        sort_idx = np.argsort(px)
        ax.plot(
            px[sort_idx],
            py[sort_idx],
            "--",
            color=anchor_color,
            lw=1.4,
            alpha=0.72,
            zorder=2,
            label="Pareto frontier",
        )
        ax.scatter(
            px,
            py,
            s=138,
            marker="D",
            color=anchor_color,
            zorder=5,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
        )

        if annotate_pareto_lambdas and not pareto_df.empty:
            pareto_df = pareto_df.sort_values("auroc")
            for i, (_, row) in enumerate(pareto_df.iterrows()):
                dx = 6
                dy = -8 if i % 2 else 8
                ax.annotate(
                    f"$\\lambda_f={row['lambda']:g}$",
                    (row["auroc"], row[y_metric]),
                    fontsize=10.5,
                    ha="left",
                    va="center",
                    xytext=(dx, dy),
                    textcoords="offset points",
                    color=anchor_color,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
                    zorder=7,
                )

    # Anchor point (lambda=0)
    if not anchors.empty:
        for _, anc in anchors.iterrows():
            ax.scatter(
                anc["auroc"],
                anc[y_metric],
                marker="*",
                s=340,
                color=anchor_color,
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
                label=r"$\lambda_f=0$ anchor",
            )

    ax.set_xlabel(r"AUROC $\uparrow$", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(series_title, pad=8)
    ax.grid(axis="y", alpha=0.15, lw=0.6)
    ax.legend(loc="best", framealpha=0.95)

    return sc

def _build_output_paths(dataset: str, output_path: Optional[Path]) -> Dict[str, Path]:
    if output_path is None:
        output_dir = PROJECT_ROOT / "results" / "paper_results"
    else:
        output_dir = output_path if output_path.suffix == "" else output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "soft_only": output_dir / f"pareto_scatter_soft_only_{dataset}.pdf",
        "soft_hard": output_dir / f"pareto_scatter_soft_hard_{dataset}.pdf",
        "csv": output_dir / f"pareto_scatter_data_{dataset}.csv",
    }


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_pareto_front(
    dataset: str,
    output_path: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
    annotate: bool = False,
    annotate_pareto: bool = True,
) -> pd.DataFrame:
    if dataset_root is None:
        dataset_root = PROJECT_ROOT / "results" / "experiments" / dataset

    config_path = (
        PROJECT_ROOT / "src" / "experiments" / "configs" / f"{dataset}_config.json"
    )
    with open(config_path, "r", encoding="utf-8") as fh:
        dataset_config = json.load(fh)
    protected_attrs: List[str] = dataset_config["fairness_settings"]["protected_attrs"]
    print(f"Dataset: {dataset} | Protected attrs: {protected_attrs}")

    print(f"Scanning {dataset_root} ...")
    all_df = scan_dataset(dataset_root, protected_attrs)

    if all_df.empty:
        raise ValueError(f"No usable results found in {dataset_root}")

    print(f"\nExtracted {len(all_df)} unique (series, lambda) points:")
    print(
        all_df[["series", "lambda", "is_anchor", "te", "pse", "overall_te", "auroc", "n_dags"]]
        .sort_values(["series", "lambda"])
        .to_string(index=False)
    )

    soft_only_df = all_df[all_df["series"] == SERIES_SOFT_ONLY].copy()
    soft_hard_df = all_df[all_df["series"] == SERIES_SOFT_HARD].copy()

    outputs = _build_output_paths(dataset, output_path)

    series_specs = [
        (
            "soft_only",
            soft_only_df,
            "Greens",
            "#2FA84F",
            "#239043",
        ),
        (
            "soft_hard",
            soft_hard_df,
            "Purples",
            "#6A51A3",
            "#54278F",
        ),
    ]

    panel_specs = [
        ("te", r"|TE| $\downarrow$", "AUROC vs |TE|"),
        ("pse", r"|PSE| $\downarrow$", "AUROC vs |PSE|"),
    ]

    for key, df, cmap, te_color, pse_color in series_specs:
        fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.1))
        sweep_df = df[(~df["is_anchor"]) & np.isfinite(df["lambda"]) & (df["lambda"] > 0)]
        shared_norm = None
        if not sweep_df.empty:
            log_vals = np.log10(sweep_df["lambda"].values)
            shared_norm = mcolors.Normalize(vmin=float(log_vals.min()), vmax=float(log_vals.max()))

        colors = [te_color, pse_color]
        first_sc = None
        for ax, panel, anchor_color in zip(axes, panel_specs, colors):
            y_metric, y_label, panel_title = panel
            sc = _draw_pareto_panel(
                ax,
                df,
                y_metric=y_metric,
                y_label=y_label,
                cmap=cmap,
                anchor_color=anchor_color,
                series_title=panel_title,
                annotate_lambdas=annotate,
                annotate_pareto_lambdas=annotate_pareto,
                color_norm=shared_norm,
            )
            if first_sc is None and sc is not None:
                first_sc = sc

        if first_sc is not None:
            cbar_ax = fig.add_axes([0.925, 0.20, 0.012, 0.62])
            cbar = fig.colorbar(
                first_sc,
                cax=cbar_ax,
                label=r"$\log_{10}(\lambda_f)$",
            )
            cbar.ax.tick_params(labelsize=10.5)

        fig.subplots_adjust(wspace=0.30, right=0.86)
        fig.savefig(outputs[key], bbox_inches="tight", dpi=150)
        print(f"\nSaved PDF: {outputs[key]}")
        plt.close(fig)

    all_df.drop(columns=["run_dir"]).sort_values(["series", "lambda"]).to_csv(outputs["csv"], index=False)
    print(f"Saved CSV: {outputs['csv']}")
    return all_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Pareto front for the lambda fairness sweep."
    )
    parser.add_argument(
        "--dataset", type=str, default="compas",
        help="Dataset name (must match a config in src/experiments/configs/).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output directory for generated PDFs. If a .pdf path is provided, "
            "its parent directory is used."
        ),
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None,
        help="Override the experiment results root directory.",
    )
    parser.add_argument(
        "--no-annotate", action="store_true",
        help="Suppress per-point lambda annotations on scatter plots (default behavior).",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Enable per-point lambda annotations for diagnostic plots.",
    )
    parser.add_argument(
        "--no-annotate-pareto", action="store_true",
        help="Disable lambda annotations for Pareto-optimal points.",
    )
    args = parser.parse_args()

    plot_pareto_front(
        dataset=args.dataset,
        output_path=Path(args.output) if args.output else None,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        annotate=args.annotate and not args.no_annotate,
        annotate_pareto=not args.no_annotate_pareto,
    )


if __name__ == "__main__":
    main()
