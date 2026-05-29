import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.compile_causal_fairness_analysis_table import (
    ConditionMetrics,
    _find_latest_run_dir,
    _fmt_bold_with_std,
    _fmt_with_std,
    _resolve_run_artifacts,
    render_latex_table,
)


class TestCompileCausalFairnessAnalysisTable(unittest.TestCase):
    def _make_condition_metrics(
        self,
        experiment: str,
        n_dags: int,
        te_mean: float,
        te_std: float,
    ) -> ConditionMetrics:
        attr_effects = {
            "race_cat": {
                "te_abs": te_mean,
                "nde_abs": te_mean,
                "pse_abs": te_mean,
                "ite_abs": te_mean,
                "violation_rate": te_mean,
                "violation_rate_pct": te_mean,
            }
        }
        attr_effects_std = {
            "race_cat": {
                "te_abs": te_std,
                "nde_abs": te_std,
                "pse_abs": te_std,
                "ite_abs": te_std,
                "violation_rate": te_std,
                "violation_rate_pct": te_std,
            }
        }
        return ConditionMetrics(
            experiment=experiment,
            run_dir=Path("."),
            result_stem=experiment,
            lambda_value=None,
            dag_id=0,
            n_dags=n_dags,
            n_edges=1,
            attr_effects=attr_effects,
            attr_effects_std=attr_effects_std,
            f1=0.5,
            f1_std=0.05,
            auroc=0.6,
            auroc_std=0.06,
            dpd_per_attr={"race_cat": te_mean},
            dpd_std_per_attr={"race_cat": te_std},
            eo_per_attr={"race_cat": te_mean},
            eo_std_per_attr={"race_cat": te_std},
        )

    def _create_result_bundle(
        self,
        dataset_root: Path,
        run_name: str,
        result_stem: str,
        config_lambda: float | None = None,
    ) -> Path:
        run_dir = dataset_root / run_name
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        (results_dir / f"{result_stem}_cf_results.csv").write_text("dag_id,data_type\n0,standard\n", encoding="utf-8")
        (results_dir / f"{result_stem}_ml_results.csv").write_text(
            "dag_id,data_type,model_type,AUROC,f1\n0,standard,xgboost,0.5,0.5\n",
            encoding="utf-8",
        )

        payload = {
            "dataset_config": {"soft_fairness_settings": {}},
            "experiment_config": {},
        }
        if config_lambda is not None:
            payload["dataset_config"]["soft_fairness_settings"]["lambda"] = config_lambda
            payload["experiment_config"]["fairness_lambda_override"] = config_lambda

        (results_dir / f"{result_stem}_config.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return run_dir

    def test_resolve_run_artifacts_accepts_lambda_suffixed_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            run_dir = self._create_result_bundle(
                dataset_root=dataset_root,
                run_name="soft_fairness_only_lambda_0p15_20260513_080639",
                result_stem="soft_fairness_only_lambda_0p15",
                config_lambda=0.15,
            )

            resolved = _resolve_run_artifacts(
                run_dir=run_dir,
                experiment="soft_fairness_only",
                lambda_filter=0.15,
            )

            self.assertEqual(resolved.result_stem, "soft_fairness_only_lambda_0p15")
            self.assertAlmostEqual(resolved.lambda_value or 0.0, 0.15)
            self.assertTrue(resolved.cf_file.exists())
            self.assertTrue(resolved.ml_file.exists())

    def test_find_latest_run_dir_handles_old_and_new_lambda_naming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            self._create_result_bundle(
                dataset_root=dataset_root,
                run_name="soft_fairness_only_20260510_140343_lambda_0p15",
                result_stem="soft_fairness_only_lambda_0p15",
                config_lambda=0.15,
            )
            newest = self._create_result_bundle(
                dataset_root=dataset_root,
                run_name="soft_fairness_only_lambda_0p15_20260513_090000",
                result_stem="soft_fairness_only_lambda_0p15",
                config_lambda=0.15,
            )
            self._create_result_bundle(
                dataset_root=dataset_root,
                run_name="soft_fairness_only_lambda_0p1_20260513_100000",
                result_stem="soft_fairness_only_lambda_0p1",
                config_lambda=0.1,
            )

            selected = _find_latest_run_dir(
                dataset_root=dataset_root,
                experiment="soft_fairness_only",
                lambda_filter=0.15,
            )
            self.assertEqual(selected, newest)

    def test_resolve_run_artifacts_uses_config_lambda_when_stem_has_no_lambda(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            run_dir = self._create_result_bundle(
                dataset_root=dataset_root,
                run_name="soft_fairness_only_20260513_090000",
                result_stem="soft_fairness_only",
                config_lambda=0.15,
            )

            resolved = _resolve_run_artifacts(
                run_dir=run_dir,
                experiment="soft_fairness_only",
                lambda_filter=0.15,
            )
            self.assertEqual(resolved.result_stem, "soft_fairness_only")
            self.assertAlmostEqual(resolved.lambda_value or 0.0, 0.15)

    def test_fmt_with_std_baseline_display(self):
        self.assertEqual(_fmt_with_std(0.1234, 0.0045, show_std=True, decimals=4), "0.1234 $\\pm$ 0.0045")
        self.assertEqual(_fmt_with_std(0.1234, 0.0045, show_std=False, decimals=4), "0.1234")

    def test_fmt_bold_with_std(self):
        self.assertEqual(
            _fmt_bold_with_std(0.1234, 0.0045, is_best=True, show_std=True, decimals=4),
            "\\textbf{0.1234 $\\pm$ 0.0045}",
        )
        self.assertEqual(
            _fmt_bold_with_std(0.1234, 0.0045, is_best=False, show_std=True, decimals=4),
            "0.1234 $\\pm$ 0.0045",
        )

    def test_render_latex_table_baseline_only_shows_uncertainty(self):
        baseline = self._make_condition_metrics(
            experiment="baseline",
            n_dags=2,
            te_mean=0.2,
            te_std=0.03,
        )
        soft_only = self._make_condition_metrics(
            experiment="soft_fairness_only",
            n_dags=1,
            te_mean=0.1,
            te_std=0.02,
        )

        latex = render_latex_table(
            dataset_name="compas",
            ordered_metrics={
                "baseline": baseline,
                "soft_fairness_only": soft_only,
            },
            protected_attrs=["race_cat"],
        )

        self.assertIn("0.2000 $\\pm$ 0.0300", latex)
        self.assertIn("0.1000", latex)
        self.assertNotIn("0.1000 $\\pm$ 0.0200", latex)

    def test_render_latex_table_domain_knowledge_multi_dag_shows_uncertainty(self):
        baseline = self._make_condition_metrics(
            experiment="baseline",
            n_dags=1,
            te_mean=0.2,
            te_std=0.03,
        )
        domain = self._make_condition_metrics(
            experiment="domain_knowledge",
            n_dags=3,
            te_mean=0.12,
            te_std=0.04,
        )

        latex = render_latex_table(
            dataset_name="law",
            ordered_metrics={
                "baseline": baseline,
                "domain_knowledge": domain,
            },
            protected_attrs=["race_cat"],
        )

        self.assertIn("\\textbf{GES + Hard}", latex)
        self.assertIn("0.1200 $\\pm$ 0.0400", latex)


if __name__ == "__main__":
    unittest.main()
