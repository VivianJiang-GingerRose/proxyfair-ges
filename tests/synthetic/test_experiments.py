import unittest
from unittest.mock import patch
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from src.experiments.run_synthetic_main_grid import build_tasks
from src.experiments.synthetic_ges_runner import run_single_experiment


class TestSyntheticExperiments(unittest.TestCase):
    @staticmethod
    def _mock_syn_payload() -> dict:
        return {
            "data": np.array(
                [
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                ],
                dtype=np.int64,
            ),
            "variable_names": ["S", "V", "Z", "Y"],
            "protected_indices": [0],
            "outcome_index": 3,
            "proxy_index": 1,
            "legit_index": 2,
            "archetype_id": 1,
            "true_edges": [],
            "true_edges_fair": [],
            "true_edges_other": [],
            "true_pse": {"total": 0.0, "direct": 0.0, "indirect_proxy": 0.0},
            "calibration": {},
            "diagnostics": {"category": {"S": {"min_category_freq": 0.5}}},
        }

    def test_all_modes_include_hard_constraints_tasks(self):
        modes = ["main_grid", "figure1", "scenario_a", "scenario_b", "ablation_ges", "result1"]
        for mode in modes:
            with self.subTest(mode=mode):
                tasks = build_tasks(smoke=True, mode=mode)
                self.assertTrue(any(task.get("algorithm") == "hard_constraints" for task in tasks))


    def test_smoke_grid_task_build(self):
        tasks = build_tasks(smoke=True)
        self.assertGreater(len(tasks), 0)
        self.assertTrue(all(task.get("enable_temporal_blacklist") is True for task in tasks))
        row = run_single_experiment(**tasks[0])
        self.assertIn("algorithm", row)


    def test_build_tasks_can_disable_temporal_blacklist(self):
        tasks = build_tasks(smoke=True, enable_temporal_blacklist=False)
        self.assertGreater(len(tasks), 0)
        self.assertTrue(all(task.get("enable_temporal_blacklist") is False for task in tasks))


    def test_displacement_task_build(self):
        tasks = build_tasks(smoke=True, mode="displacement")
        self.assertEqual(len(tasks), 12)
        conditions = {task["condition"] for task in tasks}
        self.assertEqual(conditions, {"baseline", "penalize_s1", "penalize_s2", "penalize_joint"})
        by_condition = {task["condition"]: task for task in tasks[:4]}
        self.assertNotIn("penalty_protected_indices", by_condition["baseline"])
        self.assertEqual(by_condition["penalize_s1"]["penalty_protected_indices"], [1])
        self.assertEqual(by_condition["penalize_s2"]["penalty_protected_indices"], [2])
        self.assertEqual(by_condition["penalize_joint"]["penalty_protected_indices"], [1, 2])


    def test_run_single_experiment_binary_discretization(self):
        row = run_single_experiment(
            archetype_id=1,
            N=500,
            phi_target=0.5,
            seed=44,
            algorithm="vanilla_ges",
            lambda_f=0.0,
            n_bins=2,
        )
        self.assertIn("shd", row)
        self.assertEqual(row.get("n_bins"), 2)


    def test_scenario_a_trend_proxy_load_directional(self):
        low = run_single_experiment(
            archetype_id=1,
            N=500,
            phi_target=0.1,
            seed=41,
            algorithm="fair_mec_phase1",
            lambda_f=0.5,
            n_bins=3,
        )
        high = run_single_experiment(
            archetype_id=1,
            N=500,
            phi_target=0.9,
            seed=42,
            algorithm="fair_mec_phase1",
            lambda_f=0.5,
            n_bins=3,
        )
        # At high proxy strength, fairness model should not increase proxy parent load.
        self.assertLessEqual(high["proxy_load_pa_y_lower"], 1.0)
        self.assertLessEqual(low["proxy_load_pa_y_lower"], 1.0)
        self.assertLessEqual(high["proxy_load_pa_y_lower"], high["proxy_load_pa_y_upper"])
        self.assertLessEqual(low["proxy_load_pa_y_lower"], low["proxy_load_pa_y_upper"])


    def test_high_lambda_drops_proxy_not_harder_than_zero_lambda(self):
        base = run_single_experiment(
            archetype_id=1,
            N=600,
            phi_target=0.9,
            seed=43,
            algorithm="fair_mec_phase1",
            lambda_f=0.0,
            n_bins=3,
        )
        high = run_single_experiment(
            archetype_id=1,
            N=600,
            phi_target=0.9,
            seed=43,
            algorithm="fair_mec_phase1",
            lambda_f=1.0,
            n_bins=3,
        )
        self.assertLessEqual(high["proxy_load_pa_y_upper"], base["proxy_load_pa_y_upper"] + 1e-9)


    def test_baseline_uses_bdeu_for_discrete_data(self):
        with patch("src.experiments.synthetic_ges_runner.generate_synthetic_dataset", return_value=self._mock_syn_payload()), \
            patch("src.experiments.synthetic_ges_runner.build_fairness_cache", return_value=object()), \
            patch("src.experiments.synthetic_ges_runner._extract_learned_edges", return_value=[]), \
            patch("src.experiments.synthetic_ges_runner._is_cpdag_graph", return_value=False), \
            patch("src.experiments.synthetic_ges_runner.collect_metrics", return_value={}), \
            patch("src.experiments.synthetic_ges_runner.run_ges", return_value={"G": object(), "undirected_edges": []}) as run_ges_mock:
            row = run_single_experiment(
                archetype_id=1,
                N=200,
                phi_target=0.5,
                seed=1,
                algorithm="vanilla_ges",
                lambda_f=0.0,
                n_bins=2,
                enable_temporal_blacklist=False,
            )

        self.assertEqual(run_ges_mock.call_args.kwargs["score_func"], "local_score_BDeu")
        self.assertEqual(row["score_func"], "local_score_BDeu")


    def test_temporal_and_constrained_modes_keep_bdeu_path(self):
        with patch("src.experiments.synthetic_ges_runner.generate_synthetic_dataset", return_value=self._mock_syn_payload()), \
            patch("src.experiments.synthetic_ges_runner.build_fairness_cache", return_value=object()), \
            patch("src.experiments.synthetic_ges_runner._extract_learned_edges", return_value=[]), \
            patch("src.experiments.synthetic_ges_runner._is_cpdag_graph", return_value=False), \
            patch("src.experiments.synthetic_ges_runner.collect_metrics", return_value={}), \
            patch("src.experiments.synthetic_ges_runner.run_ges", return_value={"G": object(), "undirected_edges": []}) as run_ges_mock:
            row_temporal = run_single_experiment(
                archetype_id=1,
                N=200,
                phi_target=0.5,
                seed=2,
                algorithm="vanilla_ges",
                lambda_f=0.0,
                n_bins=2,
                enable_temporal_blacklist=True,
            )
            row_hard = run_single_experiment(
                archetype_id=1,
                N=200,
                phi_target=0.5,
                seed=3,
                algorithm="hard_constraints",
                lambda_f=0.0,
                n_bins=2,
                enable_temporal_blacklist=False,
            )

        self.assertEqual(run_ges_mock.call_args_list[0].kwargs["score_func"], "local_score_BDeu")
        self.assertEqual(run_ges_mock.call_args_list[1].kwargs["score_func"], "local_score_BDeu")
        self.assertEqual(row_temporal["score_func"], "local_score_BDeu")
        self.assertEqual(row_hard["score_func"], "local_score_BDeu")


    def test_displacement_single_attribute_penalty_routing(self):
        payload = {
            "data": np.array(
                [
                    [0, 0, 1, 0, 1, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0],
                    [2, 2, 1, 2, 1, 2, 1],
                    [3, 3, 2, 3, 2, 3, 2],
                    [4, 4, 3, 4, 3, 4, 3],
                    [0, 1, 4, 0, 4, 0, 4],
                    [1, 2, 0, 1, 0, 1, 0],
                    [2, 3, 1, 2, 1, 2, 1],
                    [3, 4, 2, 3, 2, 3, 2],
                    [4, 0, 3, 4, 3, 4, 3],
                    [0, 1, 4, 0, 4, 0, 4],
                    [1, 2, 0, 1, 0, 1, 0],
                ],
                dtype=np.int64,
            ),
            "variable_names": ["C", "S1", "S2", "V1", "V2", "Z", "Y"],
            "protected_indices": [1, 2],
            "outcome_index": 6,
            "proxy_index": 3,
            "proxy_indices": [3, 4],
            "legit_index": 5,
            "legit_indices": [5],
            "archetype_id": 4,
            "true_edges": [("C", "S1"), ("C", "S2"), ("S1", "V1"), ("S2", "V2"), ("V1", "Y"), ("V2", "Y"), ("Z", "Y")],
            "true_coefficients": {},
            "true_pse": {"pse_s1": 0.5, "pse_s2": 0.5, "pse_total": 1.0},
            "displacement_paths": [("S1", "V1", "Y"), ("S2", "V2", "Y")],
            "calibration": {
                "corr_s1_s2_continuous": 0.5,
                "r2_v1_s1_continuous": 0.5,
                "r2_v2_s2_continuous": 0.5,
                "nmi_v1_s1_discrete": 0.5,
                "nmi_v2_s2_discrete": 0.5,
                "cramers_v_v1_s1_discrete": 0.5,
                "cramers_v_v2_s2_discrete": 0.5,
            },
            "diagnostics": {"category": {"C": {"min_category_freq": 2}}},
        }
        with patch("src.experiments.synthetic_ges_runner.generate_synthetic_dataset", return_value=payload), \
            patch("src.experiments.synthetic_ges_runner._extract_learned_edges", return_value=[]), \
            patch("src.experiments.synthetic_ges_runner._is_cpdag_graph", return_value=False), \
            patch("src.experiments.synthetic_ges_runner.run_ges", return_value={"G": object(), "undirected_edges": []}) as run_ges_mock:
            row = run_single_experiment(
                archetype_id=4,
                N=200,
                phi_target=0.5,
                seed=1,
                algorithm="fair_mec_full",
                lambda_f=1.0,
                n_bins=5,
                penalty_protected_indices=[1],
                condition="penalize_s1",
            )

        self.assertEqual(run_ges_mock.call_args.kwargs["protected_attr_indices"], [1])
        self.assertEqual(row["condition"], "penalize_s1")
        self.assertEqual(row["score_func"], "local_score_BDeu")


    def test_cli_smoke_writes_score_func_column_and_expected_routing(self):
        repo_root = Path(__file__).resolve().parents[2]
        results_root = repo_root / "results" / "synthetic"
        results_root.mkdir(parents=True, exist_ok=True)
        before = {p for p in results_root.glob("main_grid_*") if p.is_dir()}

        cmd = [
            sys.executable,
            "src/experiments/run_synthetic_main_grid.py",
            "--smoke",
            "--mode",
            "main_grid",
            "--n-jobs",
            "1",
            "--disable-temporal-blacklist",
        ]
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            msg = (
                "CLI smoke run failed with non-zero exit code.\n"
                f"returncode={completed.returncode}\n"
                f"stdout_tail={completed.stdout[-2000:]}\n"
                f"stderr_tail={completed.stderr[-2000:]}"
            )
            self.fail(msg)
        self.assertIn("Saved", completed.stdout)

        after = {p for p in results_root.glob("main_grid_*") if p.is_dir()}
        created = sorted(after - before)
        self.assertTrue(created, "Expected a new main_grid_* output directory")
        out_dir = created[-1]

        csv_path = out_dir / "results.csv"
        self.assertTrue(csv_path.exists(), f"Missing expected results CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        self.assertIn("score_func", df.columns)

        vanilla_scores = set(df.loc[df["algorithm"] == "vanilla_ges", "score_func"].dropna().astype(str).unique())
        self.assertEqual(vanilla_scores, {"local_score_BDeu"})

        constrained_scores = set(
            df.loc[df["algorithm"].isin(["hard_constraints", "fair_mec_phase1", "fair_mec_full", "fair_mec_marginal_only"]), "score_func"]
            .dropna()
            .astype(str)
            .unique()
        )
        self.assertEqual(constrained_scores, {"local_score_BDeu"})


if __name__ == "__main__":
    unittest.main()
