import unittest

import numpy as np
import pandas as pd

from src.experiments.statistical_tests import run_mcnemar_tests, run_wilcoxon_tests, PairSpec


class TestStatisticalTests(unittest.TestCase):
    def test_mcnemar_outputs_expected_columns(self):
        rows = []
        for seed in range(10):
            rows.append(
                {
                    "seed": seed,
                    "algorithm": "fair_mec_full",
                    "phi_target": 0.5,
                    "n_bins": 2,
                    "suppressed_proxy": 1.0 if seed < 8 else 0.0,
                    "retained_legit": 1.0,
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "algorithm": "vanilla_ges",
                    "phi_target": 0.5,
                    "n_bins": 2,
                    "suppressed_proxy": 1.0 if seed < 3 else 0.0,
                    "retained_legit": 1.0,
                }
            )

        df = pd.DataFrame(rows)
        out = run_mcnemar_tests(df=df, pairs=[PairSpec("fair_mec_full", "vanilla_ges")])
        self.assertGreater(len(out), 0)
        self.assertIn("p_value", out.columns)
        self.assertIn("p_value_bonferroni", out.columns)
        self.assertTrue(np.all((out["p_value_bonferroni"].dropna() >= 0.0) & (out["p_value_bonferroni"].dropna() <= 1.0)))

    def test_wilcoxon_outputs_expected_columns(self):
        rows = []
        for seed in range(12):
            rows.append(
                {
                    "seed": seed,
                    "algorithm": "fair_mec_full",
                    "phi_target": 0.5,
                    "n_bins": 2,
                    "shd": float(2 + (seed % 3)),
                    "pse_total_error": float(0.1 + 0.01 * seed),
                    "dp_gap": float(0.2 + 0.01 * seed),
                    "cf_violation": float(0.05 + 0.005 * seed),
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "algorithm": "hard_constraints",
                    "phi_target": 0.5,
                    "n_bins": 2,
                    "shd": float(3 + (seed % 3)),
                    "pse_total_error": float(0.12 + 0.01 * seed),
                    "dp_gap": float(0.23 + 0.01 * seed),
                    "cf_violation": float(0.06 + 0.005 * seed),
                }
            )

        df = pd.DataFrame(rows)
        out = run_wilcoxon_tests(df=df, pairs=[PairSpec("fair_mec_full", "hard_constraints")])
        self.assertGreater(len(out), 0)
        self.assertIn("p_value", out.columns)
        self.assertIn("p_value_bonferroni", out.columns)
        self.assertTrue(np.all((out["p_value_bonferroni"].dropna() >= 0.0) & (out["p_value_bonferroni"].dropna() <= 1.0)))


if __name__ == "__main__":
    unittest.main()
