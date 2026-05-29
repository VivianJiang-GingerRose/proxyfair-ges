import numpy as np
import unittest

from src.experiments.synthetic_ges_runner import (
    _algorithm_to_params,
    build_fairness_cache,
    collect_displacement_metrics,
    collect_metrics,
    run_single_experiment,
)
from src.faircausal.data.synthetic.generator import generate_synthetic_dataset


class TestSyntheticIntegration(unittest.TestCase):
    def test_hard_constraints_include_fairness_and_temporal_blocks(self):
        # Variable order for archetype 1: S, V, Z, Y
        params = _algorithm_to_params(
            algorithm="hard_constraints",
            protected_indices=[0],
            outcome_index=3,
            proxy_index=1,
            n_vars=4,
            lambda_f=0.0,
            enable_temporal_blacklist=True,
        )
        blacklist = params["blacklist_matrix"]

        # Fairness constraint: S -> Y forbidden.
        self.assertTrue(blacklist[0, 3])

        # Temporal constraints: Y is terminal.
        self.assertTrue(blacklist[3, 0])
        self.assertTrue(blacklist[3, 1])
        self.assertTrue(blacklist[3, 2])

        # Temporal constraints: S is predetermined/exogenous.
        self.assertTrue(blacklist[1, 0])
        self.assertTrue(blacklist[2, 0])
        self.assertTrue(blacklist[3, 0])

        # Temporal constraints: V can only point to Y.
        self.assertTrue(blacklist[1, 0])
        self.assertTrue(blacklist[1, 2])
        self.assertFalse(blacklist[1, 3])

        # Z -> V can remain allowed while V -> Z is blocked by temporal ordering.
        self.assertFalse(blacklist[2, 1])
        self.assertTrue(blacklist[1, 2])


    def test_temporal_blacklist_can_be_disabled(self):
        params = _algorithm_to_params(
            algorithm="fair_mec_phase1",
            protected_indices=[0],
            outcome_index=3,
            proxy_index=1,
            n_vars=4,
            lambda_f=0.1,
            enable_temporal_blacklist=False,
        )
        self.assertIsNone(params["blacklist_matrix"])


    def test_hard_constraints_keep_fairness_ban_when_temporal_disabled(self):
        params = _algorithm_to_params(
            algorithm="hard_constraints",
            protected_indices=[0],
            outcome_index=3,
            proxy_index=1,
            n_vars=4,
            lambda_f=0.0,
            enable_temporal_blacklist=False,
        )
        blacklist = params["blacklist_matrix"]

        # Fairness hard constraint remains active.
        self.assertTrue(blacklist[0, 3])

        # Temporal constraints are disabled.
        self.assertFalse(blacklist[3, 0])
        self.assertFalse(blacklist[1, 0])
        self.assertFalse(blacklist[1, 2])


    def test_run_single_baseline(self):
        row = run_single_experiment(
            archetype_id=1,
            N=300,
            phi_target=0.5,
            seed=11,
            algorithm="vanilla_ges",
            lambda_f=0.0,
            n_bins=3,
        )
        self.assertIn("shd", row)
        self.assertEqual(row["algorithm"], "vanilla_ges")


    def test_run_single_fairness(self):
        row = run_single_experiment(
            archetype_id=1,
            N=300,
            phi_target=0.9,
            seed=12,
            algorithm="fair_mec_phase1",
            lambda_f=1.0,
            n_bins=3,
        )
        self.assertIn("proxy_load_pa_y_lower", row)
        self.assertIn("proxy_load_pa_y_upper", row)
        self.assertEqual(row["algorithm"], "fair_mec_phase1")


    def test_run_single_fair_mec_full_outputs_dag_consistent_cpdag_string(self):
        row = run_single_experiment(
            archetype_id=1,
            N=300,
            phi_target=0.5,
            seed=33,
            algorithm="fair_mec_full",
            lambda_f=0.1,
            n_bins=3,
        )
        self.assertEqual(row["graph_type"], "dag")
        self.assertNotIn("--", row["final_cpdag"])


    def test_metric_collection_shape(self):
        syn = generate_synthetic_dataset(archetype_id=1, N=300, phi_target=0.5, n_bins=3, seed=13)
        cache = build_fairness_cache(
            data=syn["data"],
            variable_names=syn["variable_names"],
            protected_indices=syn["protected_indices"],
            outcome_index=syn["outcome_index"],
            tau_c=5.0,
        )
        row = collect_metrics(
            learned_edges=[("S", "V"), ("V", "Y")],
            syn=syn,
            fairness_cache=cache,
            algorithm="unit",
            lambda_f=0.1,
            seed=13,
            phi_target=0.5,
            N=300,
            n_bins=3,
        )
        expected = {
            "shd",
            "precision_fair",
            "recall_fair",
            "proxy_load_pa_y_lower",
            "proxy_load_pa_y_upper",
            "suppressed_proxy",
            "retained_legit",
            "shd_fair_critical",
            "shd_legit",
            "delta_star_vy",
            "pse_total_error",
            "mse",
        }
        self.assertTrue(expected.issubset(set(row.keys())))


    def test_collect_metrics_dag_row_has_no_undirected_tokens(self):
        syn = generate_synthetic_dataset(archetype_id=1, N=300, phi_target=0.5, n_bins=3, seed=31)
        cache = build_fairness_cache(
            data=syn["data"],
            variable_names=syn["variable_names"],
            protected_indices=syn["protected_indices"],
            outcome_index=syn["outcome_index"],
            tau_c=5.0,
        )
        row = collect_metrics(
            learned_edges=[("S", "V"), ("V", "Y")],
            syn=syn,
            fairness_cache=cache,
            algorithm="unit",
            lambda_f=0.1,
            seed=31,
            phi_target=0.5,
            N=300,
            n_bins=3,
            graph_type="dag",
            undirected_edges=[("S", "Z")],
        )
        self.assertNotIn("--", row["final_cpdag"])


    def test_collect_metrics_cpdag_row_can_include_undirected_tokens(self):
        syn = generate_synthetic_dataset(archetype_id=1, N=300, phi_target=0.5, n_bins=3, seed=32)
        cache = build_fairness_cache(
            data=syn["data"],
            variable_names=syn["variable_names"],
            protected_indices=syn["protected_indices"],
            outcome_index=syn["outcome_index"],
            tau_c=5.0,
        )
        row = collect_metrics(
            learned_edges=[("S", "V"), ("V", "Y")],
            syn=syn,
            fairness_cache=cache,
            algorithm="unit",
            lambda_f=0.1,
            seed=32,
            phi_target=0.5,
            N=300,
            n_bins=3,
            graph_type="cpdag",
            undirected_edges=[("S", "Z")],
        )
        self.assertIn("--", row["final_cpdag"])


    def test_displacement_metric_collection_paths(self):
        syn = generate_synthetic_dataset(archetype_id=4, N=1000, phi_target=0.5, n_bins=5, seed=24)
        row_full = collect_displacement_metrics(
            learned_edges=[("S1", "V1"), ("V1", "Y"), ("S2", "V2"), ("V2", "Y"), ("Z", "Y")],
            syn=syn,
            algorithm="unit",
            lambda_f=1.0,
            seed=24,
            N=1000,
            n_bins=5,
            condition="unit",
        )
        self.assertNotEqual(row_full["pse_s1"], 0.0)
        self.assertNotEqual(row_full["pse_s2"], 0.0)
        self.assertEqual(row_full["retained_z_y"], 1.0)

        row_missing_v1_y = collect_displacement_metrics(
            learned_edges=[("S1", "V1"), ("S2", "V2"), ("V2", "Y"), ("Z", "Y")],
            syn=syn,
            algorithm="unit",
            lambda_f=1.0,
            seed=24,
            N=1000,
            n_bins=5,
            condition="unit",
        )
        self.assertEqual(row_missing_v1_y["pse_s1"], 0.0)
        self.assertNotEqual(row_missing_v1_y["pse_s2"], 0.0)

        row_missing_z_y = collect_displacement_metrics(
            learned_edges=[("S1", "V1"), ("V1", "Y"), ("S2", "V2"), ("V2", "Y")],
            syn=syn,
            algorithm="unit",
            lambda_f=1.0,
            seed=24,
            N=1000,
            n_bins=5,
            condition="unit",
        )
        self.assertEqual(row_missing_z_y["retained_z_y"], 0.0)


    def test_fairness_cache_consistency(self):
        syn = generate_synthetic_dataset(archetype_id=1, N=300, phi_target=0.5, n_bins=3, seed=14)
        cache1 = build_fairness_cache(
            data=syn["data"],
            variable_names=syn["variable_names"],
            protected_indices=syn["protected_indices"],
            outcome_index=syn["outcome_index"],
            tau_c=5.0,
        )
        cache2 = build_fairness_cache(
            data=syn["data"],
            variable_names=syn["variable_names"],
            protected_indices=syn["protected_indices"],
            outcome_index=syn["outcome_index"],
            tau_c=5.0,
        )
        self.assertTrue(np.allclose(cache1.pi_sigma, cache2.pi_sigma))


    def test_guardrails_reject_float(self):
        # Internally run_single_experiment generates int data; this test targets cache path.
        syn = generate_synthetic_dataset(archetype_id=1, N=200, phi_target=0.5, n_bins=3, seed=22)
        data = syn["data"].astype(np.float64)
        if np.issubdtype(data.dtype, np.integer):
            raise AssertionError("unexpected dtype")
        # Trigger the same validation logic by passing invalid dtype through helper call path.
        from src.experiments.synthetic_ges_runner import _validate_discrete_input

        with self.assertRaises(ValueError):
            _validate_discrete_input(data)


if __name__ == "__main__":
    unittest.main()
