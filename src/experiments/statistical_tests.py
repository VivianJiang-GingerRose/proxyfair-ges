from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


BINARY_METRICS = ("suppressed_proxy", "retained_legit")
CONTINUOUS_METRICS = ("shd", "pse_total_error", "dp_gap", "cf_violation")
DEFAULT_GROUP_COLS = ("phi_target", "n_bins")


@dataclass(frozen=True)
class PairSpec:
    reference: str
    comparator: str



def _bonferroni_correct(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    out = df.copy()
    valid = out[p_col].notna()
    m = int(valid.sum())
    if m == 0:
        out["p_value_bonferroni"] = np.nan
        return out
    out["p_value_bonferroni"] = np.nan
    out.loc[valid, "p_value_bonferroni"] = np.minimum(1.0, out.loc[valid, p_col] * m)
    return out



def _aligned_pairs(
    df: pd.DataFrame,
    metric: str,
    reference: str,
    comparator: str,
    group_cols: Sequence[str],
) -> List[Tuple[Tuple[object, ...], pd.DataFrame]]:
    present_group_cols = [c for c in group_cols if c in df.columns]

    work = df[df["algorithm"].isin([reference, comparator])].copy()
    if work.empty:
        return []

    key_cols = ["seed", *present_group_cols]
    if "seed" not in work.columns:
        raise ValueError("Input dataframe must contain a 'seed' column for paired tests")

    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return []

    pivot = (
        work.pivot_table(
            index=key_cols,
            columns="algorithm",
            values=metric,
            aggfunc="mean",
        )
        .dropna(subset=[reference, comparator])
        .reset_index()
    )
    if pivot.empty:
        return []

    by_cols = present_group_cols if present_group_cols else ["__all__"]
    if by_cols == ["__all__"]:
        pivot["__all__"] = "all"

    grouped = []
    for group_key, gdf in pivot.groupby(by_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        grouped.append((group_key, gdf))
    return grouped



def _exact_mcnemar_pvalue(success_ref: np.ndarray, success_cmp: np.ndarray) -> Tuple[float, int, int]:
    both = np.column_stack([success_ref, success_cmp]).astype(int)
    b = int(np.sum((both[:, 0] == 1) & (both[:, 1] == 0)))
    c = int(np.sum((both[:, 0] == 0) & (both[:, 1] == 1)))

    n_disc = b + c
    if n_disc == 0:
        return 1.0, b, c

    p = float(binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided").pvalue)
    return p, b, c



def run_mcnemar_tests(
    df: pd.DataFrame,
    pairs: Sequence[PairSpec],
    metrics: Sequence[str] = BINARY_METRICS,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pair in pairs:
        for metric in metrics:
            if metric not in df.columns:
                continue

            grouped = _aligned_pairs(
                df=df,
                metric=metric,
                reference=pair.reference,
                comparator=pair.comparator,
                group_cols=group_cols,
            )
            for group_key, gdf in grouped:
                ref_vals = gdf[pair.reference].to_numpy(dtype=float)
                cmp_vals = gdf[pair.comparator].to_numpy(dtype=float)
                ref_success = (ref_vals >= 0.5).astype(int)
                cmp_success = (cmp_vals >= 0.5).astype(int)

                p_value, b, c = _exact_mcnemar_pvalue(ref_success, cmp_success)
                row: Dict[str, object] = {
                    "test": "mcnemar_exact",
                    "metric": metric,
                    "reference": pair.reference,
                    "comparator": pair.comparator,
                    "n_pairs": int(len(gdf)),
                    "b_ref1_cmp0": b,
                    "c_ref0_cmp1": c,
                    "p_value": p_value,
                    "ref_success_rate": float(np.mean(ref_success)) if len(ref_success) else np.nan,
                    "cmp_success_rate": float(np.mean(cmp_success)) if len(cmp_success) else np.nan,
                }
                for idx, col in enumerate([c for c in group_cols if c in df.columns]):
                    row[col] = group_key[idx]
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return _bonferroni_correct(out, p_col="p_value")



def run_wilcoxon_tests(
    df: pd.DataFrame,
    pairs: Sequence[PairSpec],
    metrics: Sequence[str] = CONTINUOUS_METRICS,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for pair in pairs:
        for metric in metrics:
            if metric not in df.columns:
                continue

            grouped = _aligned_pairs(
                df=df,
                metric=metric,
                reference=pair.reference,
                comparator=pair.comparator,
                group_cols=group_cols,
            )
            for group_key, gdf in grouped:
                ref_vals = gdf[pair.reference].to_numpy(dtype=float)
                cmp_vals = gdf[pair.comparator].to_numpy(dtype=float)
                diffs = ref_vals - cmp_vals

                if len(diffs) == 0:
                    continue

                all_zero = bool(np.allclose(diffs, 0.0))
                if all_zero:
                    p_value = 1.0
                    stat = 0.0
                else:
                    try:
                        stat, p_value = wilcoxon(ref_vals, cmp_vals, alternative="two-sided", zero_method="wilcox")
                        stat = float(stat)
                        p_value = float(p_value)
                    except ValueError:
                        stat = np.nan
                        p_value = np.nan

                row: Dict[str, object] = {
                    "test": "wilcoxon_signed_rank",
                    "metric": metric,
                    "reference": pair.reference,
                    "comparator": pair.comparator,
                    "n_pairs": int(len(diffs)),
                    "statistic": stat,
                    "p_value": p_value,
                    "ref_mean": float(np.mean(ref_vals)) if len(ref_vals) else np.nan,
                    "cmp_mean": float(np.mean(cmp_vals)) if len(cmp_vals) else np.nan,
                    "mean_diff_ref_minus_cmp": float(np.mean(diffs)) if len(diffs) else np.nan,
                }
                for idx, col in enumerate([c for c in group_cols if c in df.columns]):
                    row[col] = group_key[idx]
                rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return _bonferroni_correct(out, p_col="p_value")



def _infer_pairs(df: pd.DataFrame, reference: str, comparators: Optional[Iterable[str]]) -> List[PairSpec]:
    algs = sorted(set(df.get("algorithm", pd.Series(dtype=str)).dropna().astype(str).tolist()))
    if reference not in algs:
        raise ValueError(f"Reference algorithm '{reference}' is not present in CSV algorithms: {algs}")

    if comparators is None:
        preferred = ["vanilla_ges", "hard_constraints", "fair_mec_marginal_only", "fair_mec_phase1"]
        comps = [a for a in preferred if a in algs and a != reference]
        if not comps:
            comps = [a for a in algs if a != reference]
    else:
        comps = [c for c in comparators if c in algs and c != reference]

    return [PairSpec(reference=reference, comparator=c) for c in comps]



def run_all_tests(
    csv_path: str,
    reference: str = "fair_mec_full",
    comparators: Optional[Iterable[str]] = None,
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    pairs = _infer_pairs(df=df, reference=reference, comparators=comparators)

    mcnemar_df = run_mcnemar_tests(df=df, pairs=pairs, group_cols=group_cols)
    wilcoxon_df = run_wilcoxon_tests(df=df, pairs=pairs, group_cols=group_cols)
    return mcnemar_df, wilcoxon_df



def _save_outputs(mcnemar_df: pd.DataFrame, wilcoxon_df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mcnemar_path = output_dir / "mcnemar_results.csv"
    wilcoxon_path = output_dir / "wilcoxon_results.csv"
    summary_path = output_dir / "summary.json"

    mcnemar_df.to_csv(mcnemar_path, index=False)
    wilcoxon_df.to_csv(wilcoxon_path, index=False)

    summary = {
        "mcnemar_rows": int(len(mcnemar_df)),
        "wilcoxon_rows": int(len(wilcoxon_df)),
        "mcnemar_significant_bonferroni": int((mcnemar_df.get("p_value_bonferroni", pd.Series(dtype=float)) < 0.05).sum()) if not mcnemar_df.empty else 0,
        "wilcoxon_significant_bonferroni": int((wilcoxon_df.get("p_value_bonferroni", pd.Series(dtype=float)) < 0.05).sum()) if not wilcoxon_df.empty else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "mcnemar_csv": str(mcnemar_path),
        "wilcoxon_csv": str(wilcoxon_path),
        "summary_json": str(summary_path),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Run McNemar/Wilcoxon tests on synthetic experiment results CSV.")
    parser.add_argument("--csv", required=True, help="Path to results.csv from synthetic experiments")
    parser.add_argument("--out-dir", required=True, help="Directory to write statistical test outputs")
    parser.add_argument(
        "--reference",
        default="fair_mec_full",
        help="Reference algorithm for pairwise comparisons",
    )
    parser.add_argument(
        "--comparators",
        default="",
        help="Comma-separated comparator algorithms. If empty, infer sensible defaults from CSV.",
    )
    parser.add_argument(
        "--group-cols",
        default=",".join(DEFAULT_GROUP_COLS),
        help="Comma-separated grouping columns for stratified tests (e.g., phi_target,n_bins)",
    )
    args = parser.parse_args()

    comparators = [x.strip() for x in args.comparators.split(",") if x.strip()] if args.comparators else None
    group_cols = [x.strip() for x in args.group_cols.split(",") if x.strip()]

    mcnemar_df, wilcoxon_df = run_all_tests(
        csv_path=args.csv,
        reference=args.reference,
        comparators=comparators,
        group_cols=group_cols,
    )

    paths = _save_outputs(mcnemar_df=mcnemar_df, wilcoxon_df=wilcoxon_df, output_dir=Path(args.out_dir))

    print("Saved statistical test outputs:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
