"""
Compile a LaTeX causal-fairness analysis table from existing experiment outputs.

Important: this script does NOT run experiments.
It only reads CSV files produced by `main_runner.py`, selects runs/metrics,
and renders a publication-style LaTeX summary table.
"""

# poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset law --experiments baseline soft_fairness_only soft_fairness --output docs/law_causal_fairness_table_pse.tex
# poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset compas --experiments baseline soft_fairness_only soft_fairness --output docs/compas_causal_fairness_table_pse.tex
# poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset dutch --experiments baseline soft_fairness_only soft_fairness --output docs/dutch_causal_fairness_table_pse.tex
# poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset law --experiments baseline soft_fairness_only soft_fairness --output --run-dir baseline=results/experiments/law/baseline_20260425_151435 --run-dir soft_fairness_only=results/experiments/law/soft_fairness_only_YYYYMMDD_HHMMSS --run-dir soft_fairness=results/experiments/law/soft_fairness_YYYYMMDD_HHMMSS

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


EXPERIMENT_DISPLAY = {
    "baseline": "Baseline$^\\dagger$",
    "domain_knowledge": "GES + Hard",
    "fairness": "+Hard Constr.",
    "soft_fairness_only": "+Soft Only",
    "soft_fairness": "+Soft Fairness",
    "ges_phase2_only": "GESPhase2Only",
}

RACE_LIKE_KEYWORDS = (
    "race",
    "ethnic",
    "country_birth",
    "citizenship",
    "nationality",
    "origin",
)

GENDER_LIKE_KEYWORDS = (
    "gender",
    "sex",
    "male",
    "female",
)


@dataclass
class ConditionMetrics:
    experiment: str
    run_dir: Path
    result_stem: str
    lambda_value: Optional[float]
    dag_id: int
    n_dags: int
    n_edges: int
    attr_effects: Dict[str, Dict[str, float]]
    attr_effects_std: Dict[str, Dict[str, float]]
    f1: float
    f1_std: float
    auroc: float
    auroc_std: float
    dpd_per_attr: Dict[str, float]
    dpd_std_per_attr: Dict[str, float]
    eo_per_attr: Dict[str, float]
    eo_std_per_attr: Dict[str, float]


@dataclass
class ResolvedRunArtifacts:
    run_dir: Path
    result_stem: str
    cf_file: Path
    ml_file: Path
    config_file: Optional[Path]
    lambda_value: Optional[float]


def _parse_value_with_pm(cell: object) -> float:
    text = str(cell)
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    return float(text)


def _maybe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _parse_run_timestamp_key(run_name: str) -> Optional[Tuple[str, str]]:
    matches = re.findall(r"(\d{8})_(\d{6})", run_name)
    if not matches:
        return None
    return matches[-1]


def _is_soft_fairness_experiment(experiment: str) -> bool:
    return experiment in {"soft_fairness", "soft_fairness_only"}


def _run_name_matches_experiment(run_name: str, experiment: str) -> bool:
    if _is_soft_fairness_experiment(experiment):
        pattern = re.compile(
            rf"^{re.escape(experiment)}(?:_lambda_[^_]+_\d{{8}}_\d{{6}}|_\d{{8}}_\d{{6}}(?:_.*)?)$"
        )
        return bool(pattern.match(run_name))

    pattern = re.compile(rf"^{re.escape(experiment)}_\d{{8}}_\d{{6}}(?:_.*)?$")
    return bool(pattern.match(run_name))


def _decode_lambda_token(lambda_text: str) -> Optional[float]:
    cleaned = str(lambda_text).strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("p", ".")
    if cleaned.startswith("m"):
        cleaned = "-" + cleaned[1:]
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_lambda_from_name(name: str) -> Optional[float]:
    match = re.search(r"_lambda_([^_]+)", name)
    if not match:
        return None
    return _decode_lambda_token(match.group(1))


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _extract_lambda_from_config_payload(payload: Dict[str, Any]) -> Optional[float]:
    candidates = [
        payload.get("dataset_config", {})
        .get("soft_fairness_settings", {})
        .get("lambda"),
        payload.get("experiment_config", {}).get("fairness_lambda_override"),
        payload.get("fairness_lambda_override"),
        payload.get("lambda"),
    ]
    for val in candidates:
        parsed = _maybe_float(val)
        if parsed is not None:
            return float(parsed)
    return None


def _lambda_matches(candidate: Optional[float], expected: Optional[float], tol: float = 1e-12) -> bool:
    if expected is None:
        return True
    if candidate is None:
        return False
    return abs(candidate - expected) <= tol


def _stem_matches_experiment(stem: str, experiment: str) -> bool:
    if stem == experiment:
        return True
    if stem.startswith(f"{experiment}_"):
        return True
    return False


def _resolve_run_artifacts(
    run_dir: Path,
    experiment: str,
    lambda_filter: Optional[float] = None,
) -> ResolvedRunArtifacts:
    results_dir = run_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results folder: {results_dir}")

    candidate_entries: List[ResolvedRunArtifacts] = []
    should_track_lambda = _is_soft_fairness_experiment(experiment)

    config_files = sorted(results_dir.glob("*_config.json"))
    for config_path in config_files:
        stem = config_path.stem
        if stem.endswith("_config"):
            stem = stem[: -len("_config")]
        cf_path = results_dir / f"{stem}_cf_results.csv"
        ml_path = results_dir / f"{stem}_ml_results.csv"
        if not cf_path.exists() or not ml_path.exists():
            continue

        payload = _read_json_file(config_path)
        lambda_value = None
        if should_track_lambda:
            if payload is not None:
                lambda_value = _extract_lambda_from_config_payload(payload)
            if lambda_value is None:
                lambda_value = _extract_lambda_from_name(stem)

        candidate_entries.append(
            ResolvedRunArtifacts(
                run_dir=run_dir,
                result_stem=stem,
                cf_file=cf_path,
                ml_file=ml_path,
                config_file=config_path,
                lambda_value=lambda_value,
            )
        )

    if not candidate_entries:
        cf_files = sorted(results_dir.glob("*_cf_results.csv"))
        for cf_path in cf_files:
            stem = cf_path.stem[: -len("_cf_results")]
            ml_path = results_dir / f"{stem}_ml_results.csv"
            if not ml_path.exists():
                continue
            config_path = results_dir / f"{stem}_config.json"
            config_ref = config_path if config_path.exists() else None
            lambda_value = None
            if should_track_lambda:
                lambda_value = _extract_lambda_from_name(stem)
                if config_ref is not None:
                    payload = _read_json_file(config_ref)
                    if payload is not None:
                        parsed = _extract_lambda_from_config_payload(payload)
                        if parsed is not None:
                            lambda_value = parsed
            candidate_entries.append(
                ResolvedRunArtifacts(
                    run_dir=run_dir,
                    result_stem=stem,
                    cf_file=cf_path,
                    ml_file=ml_path,
                    config_file=config_ref,
                    lambda_value=lambda_value,
                )
            )

    if not candidate_entries:
        raise FileNotFoundError(
            f"No paired CF/ML result files found in {results_dir} for experiment='{experiment}'."
        )

    exp_candidates = [
        c for c in candidate_entries if _stem_matches_experiment(c.result_stem, experiment)
    ]
    if exp_candidates:
        candidate_entries = exp_candidates
    elif len(candidate_entries) > 1:
        raise ValueError(
            f"Multiple result stems found in {results_dir}, but none match experiment '{experiment}'. "
            f"Found stems: {[c.result_stem for c in candidate_entries]}"
        )

    lambda_candidates = [
        c for c in candidate_entries if _lambda_matches(c.lambda_value, lambda_filter)
    ]
    if lambda_filter is not None:
        if not lambda_candidates:
            raise ValueError(
                f"No result files in {results_dir} match lambda={lambda_filter} for "
                f"experiment='{experiment}'. Found lambdas: "
                f"{[c.lambda_value for c in candidate_entries]}"
            )
        candidate_entries = lambda_candidates

    if len(candidate_entries) == 1:
        return candidate_entries[0]

    # Prefer exact stem equality first.
    exact = [c for c in candidate_entries if c.result_stem == experiment]
    if len(exact) == 1:
        return exact[0]

    # Then prefer longest stem (usually most specific).
    by_stem_length = sorted(candidate_entries, key=lambda c: len(c.result_stem), reverse=True)
    if len(by_stem_length) >= 2 and len(by_stem_length[0].result_stem) != len(by_stem_length[1].result_stem):
        return by_stem_length[0]

    raise ValueError(
        f"Ambiguous result files in {results_dir} for experiment='{experiment}'. "
        f"Candidates: {[c.result_stem for c in candidate_entries]}"
    )


def _find_latest_run_dir(
    dataset_root: Path,
    experiment: str,
    lambda_filter: Optional[float] = None,
) -> Path:
    candidates: List[Tuple[Tuple[str, str], Path]] = []

    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        if not _run_name_matches_experiment(child.name, experiment):
            continue
        ts_key = _parse_run_timestamp_key(child.name)
        if ts_key is None:
            continue
        try:
            _resolve_run_artifacts(child, experiment=experiment, lambda_filter=lambda_filter)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append((ts_key, child))

    if not candidates:
        suffix = (
            f" and lambda={lambda_filter}" if lambda_filter is not None else ""
        )
        raise FileNotFoundError(
            f"No run directory found for experiment='{experiment}'{suffix} under {dataset_root}"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_dataset_config(dataset_name: str) -> Dict[str, object]:
    config_path = Path(__file__).parent / "configs" / f"{dataset_name}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_default_protected_attrs(
    dataset_config: Dict[str, object],
) -> Tuple[Optional[str], Optional[str], List[str]]:
    fairness_settings = dataset_config.get("fairness_settings", {})
    protected_attrs = list(fairness_settings.get("protected_attrs", []))
    if not protected_attrs:
        return None, None, []

    def matches_keywords(attr_name: str, keywords: Tuple[str, ...]) -> bool:
        lowered = attr_name.lower()
        return any(k in lowered for k in keywords)

    primary = next(
        (a for a in protected_attrs if matches_keywords(a, RACE_LIKE_KEYWORDS)),
        protected_attrs[0],
    )

    secondary = next(
        (
            a
            for a in protected_attrs
            if a != primary and matches_keywords(a, GENDER_LIKE_KEYWORDS)
        ),
        None,
    )
    if secondary is None:
        secondary = next((a for a in protected_attrs if a != primary), None)

    return primary, secondary, protected_attrs


def _pretty_attr_label(attr_name: str) -> str:
    return attr_name.replace("_", " ").strip().title()


def _select_dag_id(cf_df: pd.DataFrame) -> int:
    """Legacy best-DAG selector retained for backward compatibility."""
    standard = cf_df[cf_df["data_type"] == "standard"].copy()
    if standard.empty:
        raise ValueError("No rows with data_type='standard' in CF results.")

    by_dag = (
        standard.groupby("dag_id", as_index=False)
        .agg(
            overall_mean_fairness_score=("overall_mean_fairness_score", "first"),
            overall_worst_fairness_score=("overall_worst_fairness_score", "first"),
            overall_mean_macd=("overall_mean_macd", "first"),
        )
        .sort_values(
            [
                "overall_mean_fairness_score",
                "overall_worst_fairness_score",
                "overall_mean_macd",
                "dag_id",
            ]
        )
    )
    return int(by_dag.iloc[0]["dag_id"])


def _attribute_row(
    cf_df: pd.DataFrame,
    dag_id: int,
    group_name: str,
    attribute_name: str,
) -> pd.Series:
    rows = cf_df[
        (cf_df["data_type"] == "standard")
        & (cf_df["dag_id"] == dag_id)
        & (cf_df["group_name"] == group_name)
        & (cf_df["attribute_name"] == attribute_name)
        & (~cf_df["attribute_has_error"].astype(bool))
    ]
    if rows.empty:
        raise ValueError(
            f"No valid CF row for dag_id={dag_id}, group='{group_name}', attribute='{attribute_name}'."
        )
    return rows.iloc[0]


def _valid_standard_dag_ids(cf_df: pd.DataFrame) -> List[int]:
    standard = cf_df[cf_df["data_type"] == "standard"].copy()
    if standard.empty:
        raise ValueError("No rows with data_type='standard' in CF results.")
    return sorted(int(x) for x in standard["dag_id"].dropna().unique())


def _mean_abs(values: Iterable[float]) -> Optional[float]:
    vals = [abs(v) for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _mean_and_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    mean_val = float(sum(vals) / len(vals))
    if len(vals) == 1:
        return mean_val, 0.0
    variance = float(sum((v - mean_val) ** 2 for v in vals) / len(vals))
    return mean_val, float(math.sqrt(max(variance, 0.0)))


def _metric_values(frame: pd.DataFrame, col: Optional[str]) -> List[float]:
    if col is None or col not in frame.columns:
        return []
    values: List[float] = []
    for val in frame[col].tolist():
        parsed = _parse_value_with_pm(val)
        if not math.isnan(parsed):
            values.append(float(parsed))
    return values


def _extract_attribute_pse_abs(row: pd.Series, prefix: str) -> Optional[float]:
    pse_cols = [
        col
        for col in row.index
        if col.startswith(prefix) and col.endswith("_pse_score")
    ]
    values = [_maybe_float(row.get(col)) for col in pse_cols]
    return _mean_abs([v for v in values if v is not None])


def _extract_condition_metrics(
    run_dir: Path,
    experiment: str,
    protected_attrs: List[str],
    primary_attr: str,
    lambda_filter: Optional[float] = None,
) -> ConditionMetrics:
    resolved = _resolve_run_artifacts(
        run_dir=run_dir,
        experiment=experiment,
        lambda_filter=lambda_filter,
    )
    cf_file = resolved.cf_file
    ml_file = resolved.ml_file

    cf_df = pd.read_csv(cf_file)
    ml_df = pd.read_csv(ml_file)

    dag_ids = _valid_standard_dag_ids(cf_df)
    standard_rows = cf_df[cf_df["data_type"] == "standard"].copy()
    edge_values = [
        int(float(v))
        for v in standard_rows["n_edges"].dropna().unique().tolist()
    ]
    n_edges = min(edge_values) if edge_values else 0

    attr_effects: Dict[str, Dict[str, float]] = {}
    attr_effects_std: Dict[str, Dict[str, float]] = {}
    for attr in protected_attrs:
        te_vals: List[float] = []
        nde_vals: List[float] = []
        pse_vals: List[float] = []
        violation_vals: List[float] = []
        ite_vals: List[float] = []

        for dag_id in dag_ids:
            try:
                attr_row = _attribute_row(cf_df, dag_id, f"single_{attr}", attr)
            except ValueError:
                continue

            te = _maybe_float(attr_row.get(f"group_{attr}_total_effect"))
            nde = _maybe_float(attr_row.get(f"group_{attr}_nde_score"))
            violation = _maybe_float(attr_row.get("group_1_fairness_violation_rate"))
            ite = _maybe_float(attr_row.get(f"group_{attr}_ite_mean"))
            attr_pse = _extract_attribute_pse_abs(attr_row, f"group_{attr}_path_")

            if attr_pse is None:
                attr_pse = abs(te - nde) if te is not None and nde is not None else float("nan")

            if te is not None:
                te_vals.append(abs(te))
            if nde is not None:
                nde_vals.append(abs(nde))
            if attr_pse is not None and not math.isnan(attr_pse):
                pse_vals.append(float(abs(attr_pse)))
            if violation is not None:
                violation_vals.append(abs(violation))
            if ite is not None:
                ite_vals.append(abs(ite))

        te_mean, te_std = _mean_and_std(te_vals)
        nde_mean, nde_std = _mean_and_std(nde_vals)
        pse_mean, pse_std = _mean_and_std(pse_vals)
        ite_mean, ite_std = _mean_and_std(ite_vals)
        violation_mean, violation_std = _mean_and_std(violation_vals)

        attr_effects[attr] = {
            "te_abs": te_mean,
            "nde_abs": nde_mean,
            "pse_abs": pse_mean,
            "ite_abs": ite_mean,
            "violation_rate": violation_mean,
            "violation_rate_pct": 100.0 * violation_mean if not math.isnan(violation_mean) else float("nan"),
        }
        attr_effects_std[attr] = {
            "te_abs": te_std,
            "nde_abs": nde_std,
            "pse_abs": pse_std,
            "ite_abs": ite_std,
            "violation_rate": violation_std,
            "violation_rate_pct": 100.0 * violation_std if not math.isnan(violation_std) else float("nan"),
        }

    if primary_attr not in attr_effects:
        raise ValueError(
            f"Primary attribute '{primary_attr}' not available in extracted effects."
        )

    xgb_rows = ml_df[
        (ml_df["data_type"] == "standard")
        & (ml_df["model_type"] == "xgboost")
        & (ml_df["dag_id"].isin(dag_ids))
    ]
    if xgb_rows.empty:
        xgb_rows = ml_df[
            (ml_df["data_type"] == "standard") & (ml_df["model_type"] == "xgboost")
        ]
    if xgb_rows.empty:
        raise ValueError(f"No xgboost rows found in {ml_file}")

    f1_mean, f1_std = _mean_and_std(_metric_values(xgb_rows, "f1"))
    auroc_mean, auroc_std = _mean_and_std(_metric_values(xgb_rows, "AUROC"))

    dpd_per_attr: Dict[str, float] = {}
    dpd_std_per_attr: Dict[str, float] = {}
    eo_per_attr: Dict[str, float] = {}
    eo_std_per_attr: Dict[str, float] = {}
    for attr in protected_attrs:
        dpd_col = f"DPD_{attr}" if f"DPD_{attr}" in xgb_rows.columns else (
            "DPD" if attr == protected_attrs[0] and "DPD" in xgb_rows.columns else None
        )
        eo_col = f"EO_{attr}" if f"EO_{attr}" in xgb_rows.columns else (
            "EO" if attr == protected_attrs[0] and "EO" in xgb_rows.columns else None
        )

        dpd_mean, dpd_std = _mean_and_std(_metric_values(xgb_rows, dpd_col))
        eo_mean, eo_std = _mean_and_std(_metric_values(xgb_rows, eo_col))
        dpd_per_attr[attr] = dpd_mean
        dpd_std_per_attr[attr] = dpd_std
        eo_per_attr[attr] = eo_mean
        eo_std_per_attr[attr] = eo_std

    return ConditionMetrics(
        experiment=experiment,
        run_dir=run_dir,
        result_stem=resolved.result_stem,
        lambda_value=resolved.lambda_value,
        dag_id=dag_ids[0],
        n_dags=len(dag_ids),
        n_edges=n_edges,
        attr_effects=attr_effects,
        attr_effects_std=attr_effects_std,
        f1=f1_mean,
        f1_std=f1_std,
        auroc=auroc_mean,
        auroc_std=auroc_std,
        dpd_per_attr=dpd_per_attr,
        dpd_std_per_attr=dpd_std_per_attr,
        eo_per_attr=eo_per_attr,
        eo_std_per_attr=eo_std_per_attr,
    )


def _find_best(
    values: Dict[str, float], higher_is_better: bool, tol: float = 1e-12
) -> Dict[str, bool]:
    finite = {k: v for k, v in values.items() if v is not None and not math.isnan(v)}
    if not finite:
        return {k: False for k in values}
    best = max(finite.values()) if higher_is_better else min(finite.values())
    out = {}
    for k, v in values.items():
        out[k] = v is not None and not math.isnan(v) and abs(v - best) <= tol
    return out


def _fmt(value: float, decimals: int = 4) -> str:
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def _fmt_with_std(value: float, std: float, show_std: bool, decimals: int = 4) -> str:
    text = _fmt(value, decimals)
    if text == "N/A":
        return text
    if not show_std or std is None or math.isnan(std):
        return text
    return text + r" $\pm$ " + _fmt(std, decimals)


def _fmt_bold(value: float, is_best: bool, decimals: int = 4) -> str:
    text = _fmt(value, decimals)
    if is_best and text != "N/A":
        return f"\\textbf{{{text}}}"
    return text


def _fmt_bold_with_std(
    value: float,
    std: float,
    is_best: bool,
    show_std: bool,
    decimals: int = 4,
) -> str:
    text = _fmt_with_std(value, std, show_std=show_std, decimals=decimals)
    if is_best and text != "N/A":
        return f"\\textbf{{{text}}}"
    return text


def render_latex_table(
    dataset_name: str,
    ordered_metrics: Dict[str, ConditionMetrics],
    protected_attrs: List[str],
) -> str:
    keys = list(ordered_metrics.keys())
    cols = [EXPERIMENT_DISPLAY.get(k, k) for k in keys]
    n_cond = len(keys)
    n_total_cols = 3 + n_cond

    def effect_metric_dict(protected_attr: str, effect_key: str) -> Dict[str, float]:
        return {
            k: ordered_metrics[k].attr_effects.get(protected_attr, {}).get(effect_key, float("nan"))
            for k in keys
        }

    effect_best: Dict[str, Dict[str, Dict[str, bool]]] = {}
    for attr in protected_attrs:
        effect_best[attr] = {
            "te_abs": _find_best(effect_metric_dict(attr, "te_abs"), higher_is_better=False),
            "nde_abs": _find_best(effect_metric_dict(attr, "nde_abs"), higher_is_better=False),
            "pse_abs": _find_best(effect_metric_dict(attr, "pse_abs"), higher_is_better=False),
            "ite_abs": _find_best(effect_metric_dict(attr, "ite_abs"), higher_is_better=False),
            "violation_rate_pct": _find_best(
                effect_metric_dict(attr, "violation_rate_pct"), higher_is_better=False
            ),
        }

    def scalar_metric_dict(field: str) -> Dict[str, float]:
        return {k: getattr(ordered_metrics[k], field) for k in keys}

    best_f1 = _find_best(scalar_metric_dict("f1"), higher_is_better=True)
    best_auroc = _find_best(scalar_metric_dict("auroc"), higher_is_better=True)
    best_dpd = {
        attr: _find_best(
            {k: ordered_metrics[k].dpd_per_attr.get(attr, float("nan")) for k in keys},
            higher_is_better=False,
        )
        for attr in protected_attrs
    }
    best_eo = {
        attr: _find_best(
            {k: ordered_metrics[k].eo_per_attr.get(attr, float("nan")) for k in keys},
            higher_is_better=False,
        )
        for attr in protected_attrs
    }

    def _mean_across_attrs(values: Iterable[float]) -> float:
        finite_vals = [v for v in values if v is not None and not math.isnan(v)]
        if not finite_vals:
            return float("nan")
        return float(sum(finite_vals) / len(finite_vals))

    overall_effects_by_condition: Dict[str, Dict[str, float]] = {
        k: {
            "te_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects.get(attr, {}).get("te_abs", float("nan"))
                for attr in protected_attrs
            ),
            "nde_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects.get(attr, {}).get("nde_abs", float("nan"))
                for attr in protected_attrs
            ),
            "pse_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects.get(attr, {}).get("pse_abs", float("nan"))
                for attr in protected_attrs
            ),
            "ite_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects.get(attr, {}).get("ite_abs", float("nan"))
                for attr in protected_attrs
            ),
            "violation_rate_pct": _mean_across_attrs(
                ordered_metrics[k].attr_effects.get(attr, {}).get("violation_rate_pct", float("nan"))
                for attr in protected_attrs
            ),
            "dpd": _mean_across_attrs(
                ordered_metrics[k].dpd_per_attr.get(attr, float("nan"))
                for attr in protected_attrs
            ),
            "eo": _mean_across_attrs(
                ordered_metrics[k].eo_per_attr.get(attr, float("nan"))
                for attr in protected_attrs
            ),
        }
        for k in keys
    }

    overall_effects_std_by_condition: Dict[str, Dict[str, float]] = {
        k: {
            "te_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects_std.get(attr, {}).get("te_abs", float("nan"))
                for attr in protected_attrs
            ),
            "nde_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects_std.get(attr, {}).get("nde_abs", float("nan"))
                for attr in protected_attrs
            ),
            "pse_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects_std.get(attr, {}).get("pse_abs", float("nan"))
                for attr in protected_attrs
            ),
            "ite_abs": _mean_across_attrs(
                ordered_metrics[k].attr_effects_std.get(attr, {}).get("ite_abs", float("nan"))
                for attr in protected_attrs
            ),
            "violation_rate_pct": _mean_across_attrs(
                ordered_metrics[k].attr_effects_std.get(attr, {}).get("violation_rate_pct", float("nan"))
                for attr in protected_attrs
            ),
            "dpd": _mean_across_attrs(
                ordered_metrics[k].dpd_std_per_attr.get(attr, float("nan"))
                for attr in protected_attrs
            ),
            "eo": _mean_across_attrs(
                ordered_metrics[k].eo_std_per_attr.get(attr, float("nan"))
                for attr in protected_attrs
            ),
        }
        for k in keys
    }

    overall_best = {
        "te_abs": _find_best(
            {k: overall_effects_by_condition[k]["te_abs"] for k in keys},
            higher_is_better=False,
        ),
        "nde_abs": _find_best(
            {k: overall_effects_by_condition[k]["nde_abs"] for k in keys},
            higher_is_better=False,
        ),
        "pse_abs": _find_best(
            {k: overall_effects_by_condition[k]["pse_abs"] for k in keys},
            higher_is_better=False,
        ),
        "ite_abs": _find_best(
            {k: overall_effects_by_condition[k]["ite_abs"] for k in keys},
            higher_is_better=False,
        ),
        "violation_rate_pct": _find_best(
            {k: overall_effects_by_condition[k]["violation_rate_pct"] for k in keys},
            higher_is_better=False,
        ),
        "dpd": _find_best(
            {k: overall_effects_by_condition[k]["dpd"] for k in keys},
            higher_is_better=False,
        ),
        "eo": _find_best(
            {k: overall_effects_by_condition[k]["eo"] for k in keys},
            higher_is_better=False,
        ),
    }

    baseline_edges = ordered_metrics["baseline"].n_edges if "baseline" in ordered_metrics else None
    non_base_edges = [
        ordered_metrics[k].n_edges for k in keys if k != "baseline" and k in ordered_metrics
    ]
    non_base_edge_text = ""
    if non_base_edges:
        unique = sorted(set(non_base_edges))
        non_base_edge_text = (
            str(unique[0]) if len(unique) == 1 else "/".join(str(x) for x in unique)
        )

    def ev(attr: str, effect_key: str, best_map: Dict[str, bool], dec: int = 4) -> str:
        return " & ".join(
            _fmt_bold_with_std(
                ordered_metrics[k].attr_effects.get(attr, {}).get(effect_key, float("nan")),
                ordered_metrics[k].attr_effects_std.get(attr, {}).get(effect_key, float("nan")),
                best_map[k],
                show_std=(ordered_metrics[k].n_dags > 1),
                decimals=dec,
            )
            for k in keys
        )

    def gv(
        attr: str,
        per_attr_dict_field: str,
        per_attr_std_dict_field: str,
        best_map: Dict[str, bool],
        dec: int = 4,
    ) -> str:
        return " & ".join(
            _fmt_bold_with_std(
                getattr(ordered_metrics[k], per_attr_dict_field).get(attr, float("nan")),
                getattr(ordered_metrics[k], per_attr_std_dict_field).get(attr, float("nan")),
                best_map[k],
                show_std=(ordered_metrics[k].n_dags > 1),
                decimals=dec,
            )
            for k in keys
        )

    def sv(field: str, std_field: str, best_map: Dict[str, bool], dec: int = 4) -> str:
        return " & ".join(
            _fmt_bold_with_std(
                getattr(ordered_metrics[k], field),
                getattr(ordered_metrics[k], std_field),
                best_map[k],
                show_std=(ordered_metrics[k].n_dags > 1),
                decimals=dec,
            )
            for k in keys
        )

    def ov(metric_key: str, best_map: Dict[str, bool], dec: int = 4) -> str:
        return " & ".join(
            _fmt_bold_with_std(
                overall_effects_by_condition[k].get(metric_key, float("nan")),
                overall_effects_std_by_condition[k].get(metric_key, float("nan")),
                best_map[k],
                show_std=(ordered_metrics[k].n_dags > 1),
                decimals=dec,
            )
            for k in keys
        )

    edge_note = (
        f"$^\\dagger$Baseline DAG has {baseline_edges} edges; other conditions use "
        f"{non_base_edge_text} edges."
        if baseline_edges is not None
        else r"$^\dagger$Baseline corresponds to the unconstrained condition."
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{%",
        f"Causal fairness on the {dataset_name.capitalize()} dataset.",
        r"All metrics are absolute magnitudes ($\downarrow$ = better).",
        r"\textit{Causal Fairness}: TE/NDE/PSE are path-specific causal effects;",
        r"ITE is the mean individual treatment effect; CF violation rate is the",
        r"fraction of disadvantaged-group individuals whose prediction changes",
        r"under counterfactual intervention.",
        r"\textit{Group Fairness}: DPD and EO are group-fairness metrics (XGBoost).",
        r"\textbf{Bold}: best per metric (by mean). Entries show mean $\pm$ std",
        r"when multiple DAGs are available. " + edge_note,
        r"}",
        r"\label{tab:causal_fairness}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lll " + ("c" * n_cond) + r"@{}}",
        r"\toprule",
        r"\multicolumn{3}{l}{\textbf{Metric}}",
        "  & " + " & ".join([f"\\textbf{{{c}}}" for c in cols]) + r" \\",
        "  & & \\# DAGs$\\downarrow$  & "
        + " & ".join([str(ordered_metrics[k].n_dags) for k in keys])
        + r" \\",
        r"\midrule",
    ]

    causal_label = r"\multirow{5}{*}{\scriptsize\textit{Causal}}"
    group_label = r"\multirow{2}{*}{\scriptsize\textit{Group}}"

    for idx, attr in enumerate(protected_attrs):
        attr_label = (
            r"\multirow{7}{*}{\makecell[l]{\textit{"
            + _pretty_attr_label(attr)
            + r"}\\[-2pt]\scriptsize protected}}"
        )
        be = effect_best[attr]
        lines.extend([
            f"  {attr_label} & {causal_label} & $|\\mathrm{{TE}}|$ $\\downarrow$ & {ev(attr, 'te_abs', be['te_abs'])} \\\\",
            f"  & & $|\\mathrm{{NDE}}|$ $\\downarrow$ & {ev(attr, 'nde_abs', be['nde_abs'])} \\\\",
            f"  & & $|\\mathrm{{PSE}}|$ $\\downarrow$ & {ev(attr, 'pse_abs', be['pse_abs'])} \\\\",
            f"  & & $|\\mathrm{{ITE}}|$ mean $\\downarrow$ & {ev(attr, 'ite_abs', be['ite_abs'])} \\\\",
            f"  & & CF viol.\\ rate (\\%) $\\downarrow$ & {ev(attr, 'violation_rate_pct', be['violation_rate_pct'], dec=2)} \\\\",
            f"  & {group_label} & DPD $\\downarrow$ & {gv(attr, 'dpd_per_attr', 'dpd_std_per_attr', best_dpd[attr])} \\\\",
            f"  & & EO $\\downarrow$ & {gv(attr, 'eo_per_attr', 'eo_std_per_attr', best_eo[attr])} \\\\",
        ])
        if idx != len(protected_attrs) - 1:
            lines.append(f"\\cmidrule(lr){{1-{n_total_cols}}}")

    overall_label = r"\multirow{7}{*}{\makecell[l]{\textit{Overall}\\[-2pt]\scriptsize avg. attrs}}"
    lines.extend([
        f"\\cmidrule(lr){{1-{n_total_cols}}}",
        f"  {overall_label} & {causal_label} & $|\\mathrm{{TE}}|$ $\\downarrow$ & {ov('te_abs', overall_best['te_abs'])} \\\\",
        f"  & & $|\\mathrm{{NDE}}|$ $\\downarrow$ & {ov('nde_abs', overall_best['nde_abs'])} \\\\",
        f"  & & $|\\mathrm{{PSE}}|$ $\\downarrow$ & {ov('pse_abs', overall_best['pse_abs'])} \\\\",
        f"  & & $|\\mathrm{{ITE}}|$ mean $\\downarrow$ & {ov('ite_abs', overall_best['ite_abs'])} \\\\",
        f"  & & CF viol.\\ rate (\\%) $\\downarrow$ & {ov('violation_rate_pct', overall_best['violation_rate_pct'], dec=2)} \\\\",
        f"  & {group_label} & DPD $\\downarrow$ & {ov('dpd', overall_best['dpd'])} \\\\",
        f"  & & EO $\\downarrow$ & {ov('eo', overall_best['eo'])} \\\\",
    ])

    f1_v = sv("f1", "f1_std", best_f1)
    auroc_v = sv("auroc", "auroc_std", best_auroc)
    lines.extend([
        r"\midrule",
        r"\multirow{2}{*}{\textit{Predictive Perf.}} & \multicolumn{2}{l}{F1 $\uparrow$} & "
        + f1_v + r" \\",
        r"  & \multicolumn{2}{l}{AUROC $\uparrow$} & " + auroc_v + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile LaTeX causal-fairness analysis table from existing "
            "main_runner outputs (does not execute experiments)."
        )
    )
    parser.add_argument(
        "--table-config",
        default=None,
        help=(
            "Optional JSON file with table settings (dataset, experiments, run_dirs, output, etc.). "
            "CLI flags override values from this file."
        ),
    )
    parser.add_argument("--dataset", default=None, help="Dataset name (e.g., law, compas)")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Root directory containing existing run folders generated by main_runner",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment conditions to include (left-to-right columns)",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help=(
            "Optional explicit run directory mapping, format: "
            "experiment=results/experiments/law/baseline_YYYYMMDD_HHMMSS"
        ),
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=None,
        help=(
            "Optional lambda filter used for selecting soft-fairness runs during "
            "auto-discovery (when --run-dir is not provided)."
        ),
    )
    parser.add_argument(
        "--race-attr",
        default=None,
        help=(
            "Primary protected attribute for first causal block. "
            "If omitted, inferred from dataset config."
        ),
    )
    parser.add_argument(
        "--gender-attr",
        default=None,
        help=(
            "Secondary protected attribute for second causal block. "
            "If omitted, inferred from dataset config."
        ),
    )
    parser.add_argument(
        "--output",
        nargs="?",
        default=None,
        const="AUTO",
        help=(
            "Optional path to write the LaTeX table. "
            "If omitted, prints to stdout. "
            "If passed without a value (--output), writes to "
            "docs/<dataset>_causal_fairness_table_pse.tex."
        ),
    )
    return parser.parse_args()


def _load_table_config(path_text: Optional[str]) -> Dict[str, Any]:
    if not path_text:
        return {}

    config_path = Path(path_text)
    if not config_path.exists():
        raise FileNotFoundError(f"Table config JSON not found: {config_path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Table config JSON must be an object at the top level.")
    return config


def _parse_run_dir_entries(entries: List[str]) -> Dict[str, Path]:
    explicit_map: Dict[str, Path] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid --run-dir entry '{item}'. Expected experiment=path")
        exp, path_text = item.split("=", 1)
        explicit_map[exp.strip()] = Path(path_text).resolve()
    return explicit_map


def main() -> None:
    args = parse_args()
    table_config = _load_table_config(args.table_config)

    dataset_name = args.dataset or table_config.get("dataset") or "law"
    results_root_text = args.results_root or table_config.get("results_root") or "results/experiments"
    experiments = (
        args.experiments
        or table_config.get("experiments")
        or ["baseline", "soft_fairness_only", "soft_fairness"]
    )
    output_setting = args.output if args.output is not None else table_config.get("output")
    race_attr_arg = args.race_attr if args.race_attr is not None else table_config.get("race_attr")
    gender_attr_arg = args.gender_attr if args.gender_attr is not None else table_config.get("gender_attr")
    lambda_value = (
        args.lambda_value
        if args.lambda_value is not None
        else _maybe_float(table_config.get("lambda_value"))
    )

    if not isinstance(experiments, list) or not all(isinstance(x, str) for x in experiments):
        raise ValueError("'experiments' must be a list of strings in CLI/config.")

    dataset_root = Path(results_root_text) / dataset_name
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset results folder not found: {dataset_root}")

    dataset_config = _load_dataset_config(dataset_name)
    inferred_primary, inferred_secondary, protected_attrs = _infer_default_protected_attrs(
        dataset_config
    )

    primary_attr = race_attr_arg or inferred_primary
    secondary_attr = gender_attr_arg or inferred_secondary
    if primary_attr is None:
        raise ValueError(
            f"Could not infer primary protected attribute for dataset '{dataset_name}'. "
            "Pass --race-attr explicitly."
        )
    if secondary_attr is None and len(protected_attrs) > 1:
        raise ValueError(
            f"Could not infer secondary protected attribute for dataset '{dataset_name}'. "
            "Pass --gender-attr explicitly."
        )

    selected_attrs = list(protected_attrs)
    if not selected_attrs:
        selected_attrs = [primary_attr]
    if primary_attr in selected_attrs:
        selected_attrs.remove(primary_attr)
    selected_attrs.insert(0, primary_attr)
    if secondary_attr is not None:
        if secondary_attr in selected_attrs:
            selected_attrs.remove(secondary_attr)
        insert_pos = 1 if len(selected_attrs) >= 1 else 0
        selected_attrs.insert(insert_pos, secondary_attr)
    if protected_attrs:
        if primary_attr not in protected_attrs:
            print(
                f"Warning: --race-attr '{primary_attr}' not in config protected_attrs={protected_attrs}"
            )
        if secondary_attr is not None and secondary_attr not in protected_attrs:
            print(
                f"Warning: --gender-attr '{secondary_attr}' not in config protected_attrs={protected_attrs}"
            )

    explicit_map: Dict[str, Path] = {}
    config_run_dirs = table_config.get("run_dirs", {})
    if config_run_dirs:
        if not isinstance(config_run_dirs, dict):
            raise ValueError("'run_dirs' in table config must be an object mapping experiment -> path")
        explicit_map.update({str(exp): Path(path_text).resolve() for exp, path_text in config_run_dirs.items()})

    # CLI run-dir mappings take precedence over JSON config mappings.
    explicit_map.update(_parse_run_dir_entries(args.run_dir))

    ordered: Dict[str, ConditionMetrics] = {}
    for exp in experiments:
        run_dir = explicit_map.get(exp)
        exp_lambda_filter: Optional[float] = None
        if _is_soft_fairness_experiment(exp):
            exp_lambda_filter = lambda_value
            if run_dir is None and exp_lambda_filter is None:
                raise ValueError(
                    "Soft-fairness auto-discovery requires --lambda-value (or table config "
                    "'lambda_value') to avoid accidentally mixing different lambda runs. "
                    f"Experiment='{exp}'."
                )
        if run_dir is None:
            run_dir = _find_latest_run_dir(dataset_root, exp, lambda_filter=exp_lambda_filter)
        ordered[exp] = _extract_condition_metrics(
            run_dir=run_dir,
            experiment=exp,
            protected_attrs=selected_attrs,
            primary_attr=primary_attr,
            lambda_filter=exp_lambda_filter,
        )

    latex = render_latex_table(
        dataset_name,
        ordered,
        protected_attrs=selected_attrs,
    )
    output_path: Optional[Path] = None
    if output_setting == "AUTO":
        output_path = Path("docs") / f"{dataset_name}_causal_fairness_table_pse.tex"
    elif output_setting:
        output_path = Path(output_setting)

    if output_path is not None:
        out_path = output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex, encoding="utf-8")
        print(f"Wrote LaTeX table to: {out_path}")
    else:
        print(latex)

    print("\nSelected runs:")
    print(f"- primary_attr: {primary_attr}")
    if secondary_attr is not None:
        print(f"- secondary_attr: {secondary_attr}")
    print(f"- all_table_attrs: {selected_attrs}")
    for exp, m in ordered.items():
        lambda_text = (
            f", lambda={m.lambda_value:g}" if m.lambda_value is not None else ""
        )
        print(
            f"- {exp}: {m.run_dir.name} (aggregated over {m.n_dags} dags, "
            f"result_stem={m.result_stem}, example_dag_id={m.dag_id}, "
            f"n_edges={m.n_edges}{lambda_text})"
        )


if __name__ == "__main__":
    main()
