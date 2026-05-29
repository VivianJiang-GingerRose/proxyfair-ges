from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from src.experiments.statistical_tests import run_all_tests


def _aggregate_points(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    keep = [c for c in group_cols if c in df.columns]
    if not keep:
        raise ValueError("No valid grouping columns found in dataframe")
    return (
        df.groupby(keep, dropna=False, as_index=False)
        .agg(
            shd=("shd", "mean"),
            proxy_load_pa_y_lower=("proxy_load_pa_y_lower", "mean"),
            proxy_load_pa_y_upper=("proxy_load_pa_y_upper", "mean"),
        )
        .sort_values(keep)
    )


def _algorithm_label(algorithm: str) -> str:
    mapping = {
        "vanilla_ges": "Vanilla GES",
        "hard_constraints": "Hard Constraints",
        "fair_mec_phase1": "Fair-MEC Phase1",
        "fair_mec_full": "Fair-MEC Full",
        "fair_mec_marginal_only": "Fair-MEC Marginal Only",
    }
    return mapping.get(algorithm, algorithm)


def _prepare_selectivity_dataframe(
    df: pd.DataFrame,
    *,
    phi: Optional[float] = None,
    archetype: int = 1,
    N: int = 2000,
) -> pd.DataFrame:
    prepared = df.copy()
    for col in ("lambda_param", "suppressed_proxy", "retained_legit"):
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    if "archetype" in prepared.columns:
        prepared = prepared[prepared["archetype"] == archetype]
    if phi is not None and "phi_target" in prepared.columns:
        prepared = prepared[prepared["phi_target"].round(3) == round(phi, 3)]
    if "N" in prepared.columns:
        prepared = prepared[prepared["N"] == N]

    prepared = prepared.dropna(subset=["lambda_param", "suppressed_proxy", "retained_legit"])
    return prepared


def _aggregate_selectivity_points(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    dfa = df[df["algorithm"] == algorithm]
    if dfa.empty:
        return pd.DataFrame(columns=["lambda_param", "tpr_fair", "tpr_legit"])
    return (
        dfa.groupby("lambda_param", as_index=False)
        .agg(tpr_fair=("suppressed_proxy", "mean"), tpr_legit=("retained_legit", "mean"))
        .sort_values("lambda_param")
    )


def _draw_selectivity_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    phi: float,
    lambda_norm: Optional[Normalize],
    cmap_name: str,
) -> List[str]:
    warnings: List[str] = []

    fair_algs = ["fair_mec_full", "fair_mec_marginal_only"]
    fair_markers: Dict[str, str] = {
        "fair_mec_full": "o",
        "fair_mec_marginal_only": "s",
    }
    baseline_algs = ["vanilla_ges", "hard_constraints"]
    baseline_colors: Dict[str, str] = {
        "vanilla_ges": "#1f77b4",
        "hard_constraints": "#ff7f0e",
    }

    for alg in fair_algs:
        agg = _aggregate_selectivity_points(df, alg)
        if agg.empty:
            warnings.append(f"No rows for algorithm '{alg}' at phi={phi}")
            continue

        lambdas = agg["lambda_param"].to_numpy()
        if lambda_norm is None:
            normalized = np.full(lambdas.shape, 0.5)
        else:
            normalized = lambda_norm(lambdas)
        alphas = 0.2 + 0.75 * normalized
        colors = plt.get_cmap(cmap_name)(normalized)
        colors[:, 3] = alphas

        ax.scatter(
            agg["tpr_fair"],
            agg["tpr_legit"],
            s=62,
            c=colors,
            marker=fair_markers.get(alg, "o"),
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            label=_algorithm_label(alg),
        )

    for alg in baseline_algs:
        dfa = df[df["algorithm"] == alg]
        if dfa.empty:
            continue
        tpr_fair = float(dfa["suppressed_proxy"].mean())
        tpr_legit = float(dfa["retained_legit"].mean())
        ax.scatter(
            [tpr_fair],
            [tpr_legit],
            marker="X",
            s=108,
            color=baseline_colors.get(alg, "#333333"),
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label=_algorithm_label(alg),
        )

    ax.set_title(f"Target proxy strength φ*={phi}", fontsize=10.8, pad=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.set_facecolor("#fbfcfe")

    return warnings


def _build_selectivity_legend_handles() -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label="Fair-MEC Full",
            markerfacecolor="#2f4b7c",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            label="Fair-MEC Marginal Only",
            markerfacecolor="#2f4b7c",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            label="Vanilla GES",
            markerfacecolor="#1f77b4",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8.5,
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="None",
            label="Hard Constraints",
            markerfacecolor="#ff7f0e",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8.5,
        ),
    ]


def plot_tradeoff(
    csv_path: str,
    output_path: str,
    fair_algorithm: str = "fair_mec_full",
) -> Tuple[str, List[str]]:
    df = pd.read_csv(csv_path)

    required = {
        "archetype",
        "algorithm",
        "lambda_param",
        "shd",
        "proxy_load_pa_y_lower",
        "proxy_load_pa_y_upper",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["lambda_param"] = pd.to_numeric(df["lambda_param"], errors="coerce")
    df["shd"] = pd.to_numeric(df["shd"], errors="coerce")
    df["proxy_load_pa_y_lower"] = pd.to_numeric(df["proxy_load_pa_y_lower"], errors="coerce")
    df["proxy_load_pa_y_upper"] = pd.to_numeric(df["proxy_load_pa_y_upper"], errors="coerce")
    df = df.dropna(subset=["lambda_param", "shd", "proxy_load_pa_y_lower", "proxy_load_pa_y_upper"])

    archetypes = sorted(df["archetype"].unique().tolist())
    n_panels = len(archetypes)
    n_cols = min(2, max(1, n_panels))
    n_rows = int(math.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    warnings: List[str] = []

    baseline_algs = ["vanilla_ges", "hard_constraints"]
    baseline_colors: Dict[str, str] = {
        "vanilla_ges": "#1f77b4",
        "hard_constraints": "#ff7f0e",
    }

    for panel_idx, archetype in enumerate(archetypes):
        ax = axes_flat[panel_idx]
        dfa = df[df["archetype"] == archetype]

        fair = dfa[dfa["algorithm"] == fair_algorithm]
        fair_agg = _aggregate_points(fair, ["lambda_param"]) if not fair.empty else pd.DataFrame()

        if fair_agg.empty:
            warnings.append(f"Archetype {archetype}: no rows for fair algorithm '{fair_algorithm}'")
        else:
            ax.plot(
                fair_agg["shd"],
                fair_agg["proxy_load_pa_y_lower"],
                marker="o",
                linewidth=2,
                color="#d62728",
                label=f"{_algorithm_label(fair_algorithm)} curve",
            )
            for _, row in fair_agg.iterrows():
                lam = row["lambda_param"]
                ax.annotate(
                    f"λ={lam:g}",
                    (row["shd"], row["proxy_load_pa_y_lower"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

            if fair_agg["lambda_param"].nunique() < 2:
                warnings.append(
                    f"Archetype {archetype}: only {fair_agg['lambda_param'].nunique()} lambda value for {fair_algorithm}; curve is not fully resolved"
                )

        for alg in baseline_algs:
            dfi = dfa[dfa["algorithm"] == alg]
            if dfi.empty:
                continue
            b = _aggregate_points(dfi, ["algorithm"]).iloc[0]
            lo = b["proxy_load_pa_y_lower"]
            hi = b["proxy_load_pa_y_upper"]
            ax.errorbar(
                [b["shd"]],
                [lo],
                yerr=[[0.0], [max(0.0, hi - lo)]],
                fmt="X",
                markersize=8,
                capsize=6,
                capthick=1.5,
                color=baseline_colors.get(alg, "black"),
                label=f"{_algorithm_label(alg)} [lower, upper]",
                zorder=5,
            )

        ax.set_title(f"Archetype {archetype}")
        ax.set_xlabel("SHD (lower is better)")
        ax.set_ylabel("Proxy Load of Pa(Y) - interval for CPDAGs, exact for Fair-MEC")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "GES Fidelity-Fairness Trade-Off\nBars = CPDAG uncertainty; curve = Fair-MEC committed orientation",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path, warnings


def plot_selectivity_curve(
    csv_path: str,
    output_path: str,
    phi: Optional[float] = None,
    N: int = 2000,
    archetype: int = 1,
) -> Tuple[str, List[str]]:
    """Plot TPR_fair vs TPR_legit selectivity curve.

    X-axis: TPR_fair = mean(suppressed_proxy) — fraction of runs where V→Y was dropped.
    Y-axis: TPR_legit = mean(retained_legit)  — fraction of runs where Z→Y was kept.

    Fair-MEC variants appear as lambda-parameterised curves; baselines as fixed scatter points.
    """
    df = pd.read_csv(csv_path)

    required = {"algorithm", "lambda_param", "suppressed_proxy", "retained_legit"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _prepare_selectivity_dataframe(df, phi=phi, archetype=archetype, N=N)

    warnings: List[str] = []

    fair_points = df[df["algorithm"].isin(["fair_mec_full", "fair_mec_marginal_only"])]["lambda_param"]
    lambda_norm: Optional[Normalize] = None
    if not fair_points.empty:
        lambda_norm = Normalize(vmin=float(fair_points.min()), vmax=float(fair_points.max()))

    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    warnings.extend(_draw_selectivity_panel(ax, df, phi=phi if phi is not None else -1.0, lambda_norm=lambda_norm, cmap_name="viridis"))

    title_parts = [f"Selectivity Curve (archetype {archetype}"]
    if phi is not None:
        title_parts.append(f", φ*={phi}")
    title_parts.append(")")
    ax.set_title("".join(title_parts), fontsize=11)
    ax.set_xlabel("TPR$_{\\mathrm{fair}}$ — P(V→Y suppressed)")
    ax.set_ylabel("TPR$_{\\mathrm{legit}}$ — P(Z→Y retained)")

    legend_handles = _build_selectivity_legend_handles()
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.95)

    if lambda_norm is not None:
        sm = ScalarMappable(norm=lambda_norm, cmap=plt.get_cmap("viridis"))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.055, pad=0.03)
        cbar.set_label("Fairness penalty λ", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return output_path, warnings


def plot_selectivity_triptych_pdf(
    df: pd.DataFrame,
    output_path: str,
    phi_values: Optional[List[float]] = None,
    archetype: int = 1,
    N: int = 2000,
) -> Tuple[str, List[str]]:
    """Render one horizontal 3-panel selectivity figure and save as PDF."""
    if phi_values is None:
        phi_values = [0.1, 0.5, 0.9]

    base_df = _prepare_selectivity_dataframe(df, phi=None, archetype=archetype, N=N)
    warnings: List[str] = []

    fair_points = base_df[base_df["algorithm"].isin(["fair_mec_full", "fair_mec_marginal_only"])]["lambda_param"]
    lambda_norm: Optional[Normalize] = None
    if not fair_points.empty:
        lambda_norm = Normalize(vmin=float(fair_points.min()), vmax=float(fair_points.max()))

    fig, axes = plt.subplots(1, len(phi_values), figsize=(15.2, 4.9), sharex=True, sharey=True)
    if len(phi_values) == 1:
        axes = [axes]

    for idx, (ax, phi) in enumerate(zip(axes, phi_values)):
        phi_df = _prepare_selectivity_dataframe(base_df, phi=phi, archetype=archetype, N=N)
        if phi_df.empty:
            warnings.append(f"No selectivity rows available for phi={phi}")
        warnings.extend(_draw_selectivity_panel(ax, phi_df, phi=phi, lambda_norm=lambda_norm, cmap_name="viridis"))
        ax.set_xlabel("TPR$_{\\mathrm{fair}}$")
        if idx == 0:
            ax.set_ylabel("TPR$_{\\mathrm{legit}}$")

    fig.suptitle("Selectivity Landscape Across Proxy Strength Levels", fontsize=13, y=0.99)

    legend_handles = _build_selectivity_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.025),
        fontsize=9,
    )

    if lambda_norm is not None:
        sm = ScalarMappable(norm=lambda_norm, cmap=plt.get_cmap("viridis"))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location="right", pad=0.015, shrink=0.92)
        cbar.set_label("Fairness penalty λ (darker and less transparent means higher λ)", fontsize=9.2)

    fig.subplots_adjust(left=0.06, right=0.9, top=0.88, bottom=0.2, wspace=0.16)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path, warnings


def _algorithm_config_label(algorithm: str, lambda_param: float) -> str:
    if algorithm == "vanilla_ges":
        return "Vanilla GES"
    if algorithm == "hard_constraints":
        return "Hard Constraints"
    if algorithm == "fair_mec_phase1":
        return "Soft-only (Phase 1)"
    if algorithm == "fair_mec_full":
        return f"Full Pipeline (lambda={lambda_param:g})"
    return f"{_algorithm_label(algorithm)} (lambda={lambda_param:g})"


def _config_sort_key(algorithm: str, lambda_param: float) -> Tuple[int, float]:
    if algorithm == "vanilla_ges":
        return (0, 0.0)
    if algorithm == "hard_constraints":
        return (1, 0.0)
    if algorithm == "fair_mec_phase1":
        return (2, float(lambda_param))
    if algorithm == "fair_mec_full":
        return (3, float(lambda_param))
    return (99, float(lambda_param))


def _resolve_metric_series(df: pd.DataFrame, metric_name: str) -> pd.Series:
    if metric_name == "te":
        if "cf_te" in df.columns:
            return df["cf_te"].abs()
        return df["est_te"].abs()
    if metric_name == "pse":
        if "cf_nie" in df.columns:
            return df["cf_nie"].abs()
        return df["est_ie_proxy"].abs()
    if metric_name == "nde":
        if "cf_nde" in df.columns:
            return df["cf_nde"].abs()
        return df["nde"].abs()
    if metric_name == "dpd":
        return df["dp_gap_lr"].abs()
    if metric_name == "eo":
        if "eo_gap" in df.columns:
            return df["eo_gap"].abs()
        return df[["tpr_gap", "fpr_gap"]].abs().max(axis=1)
    if metric_name == "auroc":
        return df["auc_roc"]
    raise ValueError(f"Unsupported metric: {metric_name}")


def _paper_phi_metric_specs(metrics: List[str]) -> List[Tuple[str, str, str]]:
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    specs = {
        "te": r"$|\mathbf{TE}|$ $\downarrow$",
        "pse": r"$|\mathbf{PSE}|$ $\downarrow$",
        "nde": r"$|\mathbf{NDE}|$ $\downarrow$",
        "auroc": r"$\mathbf{AUROC}$ $\uparrow$",
    }
    unknown = [m for m in metrics if m not in specs]
    if unknown:
        raise ValueError(f"Unsupported paper phi-sweep metrics: {unknown}")
    return [(metric, specs[metric], labels[idx]) for idx, metric in enumerate(metrics)]


def _paper_phi_label(algorithm: str, lambda_param: float) -> str:
    if algorithm == "vanilla_ges":
        return "Baseline"
    if algorithm == "hard_constraints":
        return "Hard"
    if algorithm == "fair_mec_phase1":
        return "Soft-only"
    if algorithm == "fair_mec_full":
        return rf"ProxyFair $\lambda_f={lambda_param:g}$"
    return _algorithm_config_label(algorithm, lambda_param)


def _paper_phi_style(algorithm: str, lambda_param: float) -> Dict[str, Any]:
    if algorithm == "vanilla_ges":
        return {"color": "#111111", "linestyle": "-", "marker": "o", "alpha": 0.94}
    if algorithm == "hard_constraints":
        return {"color": "#E65100", "linestyle": "--", "marker": "s", "alpha": 0.92}
    if algorithm == "fair_mec_phase1":
        return {"color": "#2FA84F", "linestyle": "-.", "marker": "^", "alpha": 0.95}

    proxyfair_colors = {
        0.1: "#9ECAE1",
        0.5: "#4292C6",
        1.0: "#08519C",
        2.0: "#08306B",
    }
    rounded = round(float(lambda_param), 4)
    color = proxyfair_colors.get(rounded, "#0F766E")
    return {"color": color, "linestyle": "-", "marker": "o", "alpha": 0.9}


def plot_phi_sweep_paper_row(
    csv_path: str,
    output_path: str,
    archetype: int = 1,
    N: int = 5000,
    metrics: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """Render the compact IEEE two-column phi-sweep figure as a vector PDF."""
    if metrics is None:
        metrics = ["te", "pse", "auroc"]

    df = pd.read_csv(csv_path)
    warnings: List[str] = []
    required = {"algorithm", "lambda_param", "phi_star", "seed", "archetype", "N"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for paper phi-sweep plot: {missing}")

    dff = df.copy()
    dff["phi_star"] = pd.to_numeric(dff["phi_star"], errors="coerce")
    dff["lambda_param"] = pd.to_numeric(dff["lambda_param"], errors="coerce")
    dff = dff[(dff["archetype"] == archetype) & (dff["N"] == N)]
    dff = dff.dropna(subset=["phi_star", "lambda_param"])
    if dff.empty:
        raise ValueError(f"No rows available for archetype={archetype}, N={N}")

    allowed = {
        ("vanilla_ges", 0.0),
        ("hard_constraints", 0.0),
        ("fair_mec_full", 0.1),
        ("fair_mec_full", 0.5),
        ("fair_mec_full", 1.0),
        ("fair_mec_full", 2.0),
    }
    dff = dff[
        dff.apply(
            lambda row: (str(row["algorithm"]), round(float(row["lambda_param"]), 4)) in allowed,
            axis=1,
        )
    ]
    if dff.empty:
        raise ValueError("No rows matched the paper phi-sweep algorithm/lambda set")

    configs = (
        dff[["algorithm", "lambda_param"]]
        .drop_duplicates()
        .to_dict("records")
    )
    configs = sorted(configs, key=lambda x: _config_sort_key(str(x["algorithm"]), float(x["lambda_param"])))
    metric_specs = _paper_phi_metric_specs(metrics)

    old_rc = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "semibold",
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    try:
        n_panels = len(metric_specs)
        width = 7.16 if n_panels >= 4 else 6.65
        fig, axes = plt.subplots(1, n_panels, figsize=(width, 2.18), squeeze=False)
        axes_flat = axes.flatten()
        handles: List[Line2D] = []
        labels: List[str] = []

        for ax, (metric_name, y_label, panel_label) in zip(axes_flat, metric_specs):
            try:
                dff_metric = dff.copy()
                dff_metric["metric_value"] = pd.to_numeric(
                    _resolve_metric_series(dff_metric, metric_name),
                    errors="coerce",
                )
            except Exception as exc:
                warnings.append(f"Skipping metric {metric_name}: {exc}")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue

            dff_metric = dff_metric.dropna(subset=["metric_value"])
            if dff_metric.empty:
                warnings.append(f"No valid values for metric {metric_name}")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue

            for cfg in configs:
                algorithm = str(cfg["algorithm"])
                lambda_param = float(cfg["lambda_param"])
                cfg_df = dff_metric[
                    (dff_metric["algorithm"] == algorithm)
                    & (dff_metric["lambda_param"].round(8) == round(lambda_param, 8))
                ]
                if cfg_df.empty:
                    continue

                agg = (
                    cfg_df.groupby("phi_star", as_index=False)
                    .agg(mean=("metric_value", "mean"), std=("metric_value", "std"))
                    .sort_values("phi_star")
                )
                x = agg["phi_star"].to_numpy(dtype=float)
                y = agg["mean"].to_numpy(dtype=float)
                style = _paper_phi_style(algorithm, lambda_param)
                line = ax.plot(
                    x,
                    y,
                    marker=style["marker"],
                    markersize=2.35,
                    linewidth=0.95,
                    linestyle=style["linestyle"],
                    color=style["color"],
                    alpha=style["alpha"],
                    markeredgecolor="white",
                    markeredgewidth=0.25,
                    label=_paper_phi_label(algorithm, lambda_param),
                    zorder=3,
                )[0]
                if metric_name == metric_specs[0][0]:
                    handles.append(line)
                    labels.append(_paper_phi_label(algorithm, lambda_param))

            ax.set_title(f"{panel_label} {y_label}", loc="left", pad=2.0, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(axis="y", color="#D1D5DB", alpha=0.45, linewidth=0.45)
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
            ax.set_facecolor("#FFFFFF")
            ax.tick_params(axis="both", length=2.0, width=0.5, pad=1.4)
            ax.spines["left"].set_color("#D1D5DB")
            ax.spines["bottom"].set_color("#D1D5DB")
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)
            ax.margins(x=0.03)

        fig.supxlabel(r"Proxy strength $\phi^\ast$", fontsize=8.1, fontweight="bold", y=0.07)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=6,
                frameon=False,
                columnspacing=0.72,
                handlelength=1.35,
                handletextpad=0.28,
            )
        fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.25, wspace=0.28)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    finally:
        plt.rcParams.update(old_rc)

    return output_path, warnings


def plot_phi_sweep_fairness_lines(
    csv_path: str,
    output_dir: str,
    archetype: int = 1,
    N: int = 5000,
) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    warnings: List[str] = []
    required = {"algorithm", "lambda_param", "phi_star", "seed", "archetype", "N"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for phi sweep plots: {missing}")

    dff = df.copy()
    dff["phi_star"] = pd.to_numeric(dff["phi_star"], errors="coerce")
    dff["lambda_param"] = pd.to_numeric(dff["lambda_param"], errors="coerce")
    dff = dff[(dff["archetype"] == archetype) & (dff["N"] == N)]
    dff = dff.dropna(subset=["phi_star", "lambda_param"])
    if dff.empty:
        return [], [f"No rows available for archetype={archetype}, N={N}"]

    metric_specs: List[Tuple[str, str, str]] = [
        ("te", "|TE|", "phi_sweep_te.png"),
        ("pse", "|PSE|", "phi_sweep_pse.png"),
        ("nde", "|NDE|", "phi_sweep_nde.png"),
        ("dpd", "|DPD|", "phi_sweep_dpd.png"),
        ("eo", "|EO|", "phi_sweep_eo.png"),
        ("auroc", "AUROC", "phi_sweep_auroc.png"),
    ]

    os.makedirs(output_dir, exist_ok=True)
    outputs: List[str] = []
    configs = (
        dff[["algorithm", "lambda_param"]]
        .drop_duplicates()
        .sort_values(["algorithm", "lambda_param"])
        .to_dict("records")
    )
    configs = sorted(configs, key=lambda x: _config_sort_key(str(x["algorithm"]), float(x["lambda_param"])))
    color_map = plt.get_cmap("tab10")

    for metric_name, y_label, file_name in metric_specs:
        try:
            dff_metric = dff.copy()
            dff_metric["metric_value"] = pd.to_numeric(_resolve_metric_series(dff_metric, metric_name), errors="coerce")
        except Exception as exc:
            warnings.append(f"Skipping metric {metric_name}: {exc}")
            continue

        dff_metric = dff_metric.dropna(subset=["metric_value"])
        if dff_metric.empty:
            warnings.append(f"No valid values for metric {metric_name}")
            continue

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        for idx, cfg in enumerate(configs):
            algorithm = str(cfg["algorithm"])
            lambda_param = float(cfg["lambda_param"])
            cfg_df = dff_metric[
                (dff_metric["algorithm"] == algorithm)
                & (dff_metric["lambda_param"].round(8) == round(lambda_param, 8))
            ]
            if cfg_df.empty:
                continue

            agg = (
                cfg_df.groupby("phi_star", as_index=False)
                .agg(mean=("metric_value", "mean"), std=("metric_value", "std"))
                .sort_values("phi_star")
            )
            x = agg["phi_star"].to_numpy(dtype=float)
            y = agg["mean"].to_numpy(dtype=float)
            std = agg["std"].fillna(0.0).to_numpy(dtype=float)
            color = color_map(idx % 10)
            label = _algorithm_config_label(algorithm, lambda_param)
            ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.18)

        ax.set_title(f"Phi-sweep fairness: {y_label}")
        ax.set_xlabel("phi*")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out_path = os.path.join(output_dir, file_name)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        outputs.append(out_path)

    return outputs, warnings


def plot_modal_graph_frequency_heatmap(
    csv_path: str,
    output_path: str,
    archetype: int = 1,
    N: int = 5000,
) -> Tuple[str, List[str]]:
    df = pd.read_csv(csv_path)
    warnings: List[str] = []
    required = {"algorithm", "lambda_param", "phi_star", "final_dag", "archetype", "N"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for modal heatmap: {missing}")

    dff = df.copy()
    dff["phi_star"] = pd.to_numeric(dff["phi_star"], errors="coerce")
    dff["lambda_param"] = pd.to_numeric(dff["lambda_param"], errors="coerce")
    dff = dff[(dff["archetype"] == archetype) & (dff["N"] == N)]
    dff = dff.dropna(subset=["phi_star", "lambda_param", "final_dag"])
    if dff.empty:
        raise ValueError(f"No rows available for archetype={archetype}, N={N}")

    grouped = dff.groupby(["algorithm", "lambda_param", "phi_star"], dropna=False)
    rows: List[Dict[str, Any]] = []
    for (algorithm, lambda_param, phi_star), grp in grouped:
        dag_counts = grp["final_dag"].value_counts()
        if dag_counts.empty:
            continue
        modal_freq = 100.0 * float(dag_counts.iloc[0]) / float(len(grp))
        rows.append(
            {
                "algorithm": str(algorithm),
                "lambda_param": float(lambda_param),
                "phi_star": float(phi_star),
                "modal_freq_pct": float(modal_freq),
                "config_label": _algorithm_config_label(str(algorithm), float(lambda_param)),
            }
        )

    if not rows:
        raise ValueError("No modal graph rows were computed")

    modal_df = pd.DataFrame(rows)
    phi_values = sorted(modal_df["phi_star"].unique().tolist())
    config_order = (
        modal_df[["algorithm", "lambda_param", "config_label"]]
        .drop_duplicates()
        .sort_values(["algorithm", "lambda_param"])
    )
    config_order = config_order.assign(
        _sort_key=config_order.apply(lambda r: _config_sort_key(str(r["algorithm"]), float(r["lambda_param"])), axis=1)
    ).sort_values("_sort_key")
    labels = config_order["config_label"].tolist()

    heat = np.full((len(labels), len(phi_values)), np.nan, dtype=float)
    label_to_row = {label: i for i, label in enumerate(labels)}
    phi_to_col = {float(phi): j for j, phi in enumerate(phi_values)}
    for _, row in modal_df.iterrows():
        r_idx = label_to_row.get(str(row["config_label"]))
        c_idx = phi_to_col.get(float(row["phi_star"]))
        if r_idx is None or c_idx is None:
            continue
        heat[r_idx, c_idx] = float(row["modal_freq_pct"])

    fig_h = max(4.8, 0.52 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0.0, vmax=100.0)
    ax.set_xticks(np.arange(len(phi_values)))
    ax.set_xticklabels([f"{phi:g}" for phi in phi_values], rotation=0)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("phi*")
    ax.set_ylabel("Algorithm configuration")
    ax.set_title("Modal graph frequency (%)")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isnan(heat[i, j]):
                continue
            text_color = "white" if heat[i, j] >= 50.0 else "black"
            ax.text(j, i, f"{heat[i, j]:.0f}", ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Modal frequency (%)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot synthetic Figure 1 trade-off curve from results CSV.")
    parser.add_argument(
        "--csv",
        default="results/synthetic/main_grid_20260406_113122/results.csv",
        help="Path to synthetic results CSV",
    )
    parser.add_argument(
        "--out",
        default="results/synthetic/figures/figure1_tradeoff_curve.png",
        help="Output image path (.png)",
    )
    parser.add_argument(
        "--fair-algorithm",
        default="fair_mec_full",
        choices=["fair_mec_phase1", "fair_mec_full"],
        help="Which Fair-MEC variant to use as the lambda curve",
    )
    parser.add_argument(
        "--run-stats",
        action="store_true",
        help="Also run McNemar and Wilcoxon tests and save CSV outputs.",
    )
    parser.add_argument(
        "--stats-out-dir",
        default="",
        help="Output directory for statistical test CSVs (default: alongside --out).",
    )
    parser.add_argument(
        "--stats-reference",
        default="fair_mec_full",
        help="Reference algorithm for pairwise statistical comparisons.",
    )
    parser.add_argument(
        "--stats-comparators",
        default="",
        help="Comma-separated comparator algorithms. Empty means auto-infer.",
    )
    parser.add_argument(
        "--stats-group-cols",
        default="phi_target,n_bins",
        help="Comma-separated columns for stratified statistical tests.",
    )
    args = parser.parse_args()

    out, warnings = plot_tradeoff(csv_path=args.csv, output_path=args.out, fair_algorithm=args.fair_algorithm)
    print(f"Saved figure to {out}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")

    if args.run_stats:
        comparators = [x.strip() for x in args.stats_comparators.split(",") if x.strip()] if args.stats_comparators else None
        group_cols = [x.strip() for x in args.stats_group_cols.split(",") if x.strip()]
        mcnemar_df, wilcoxon_df = run_all_tests(
            csv_path=args.csv,
            reference=args.stats_reference,
            comparators=comparators,
            group_cols=group_cols,
        )

        if args.stats_out_dir:
            stats_out_dir = args.stats_out_dir
        else:
            stats_out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "stats")

        os.makedirs(stats_out_dir, exist_ok=True)
        mcnemar_path = os.path.join(stats_out_dir, "mcnemar_results.csv")
        wilcoxon_path = os.path.join(stats_out_dir, "wilcoxon_results.csv")
        mcnemar_df.to_csv(mcnemar_path, index=False)
        wilcoxon_df.to_csv(wilcoxon_path, index=False)
        print(f"Saved McNemar results to {mcnemar_path}")
        print(f"Saved Wilcoxon results to {wilcoxon_path}")


if __name__ == "__main__":
    main()
