import unittest

import numpy as np

from src.experiments.compile_causal_fairness_analysis_table import EXPERIMENT_DISPLAY
from src.experiments.run_ges_phase2_only import (
    ScoredDag,
    _compute_proxy_load,
    _score_dags_by_phase2,
    _summarize_scored_dags,
)


class TestGesPhase2Only(unittest.TestCase):
    def test_fully_oriented_metadata_has_equal_scores(self):
        node_names = ["S", "Y"]
        data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        proxy_load = _compute_proxy_load(data, node_names, [0])

        scored = _score_dags_by_phase2(
            [[("S", "Y")]],
            data=data,
            node_names=node_names,
            proxy_load=proxy_load,
            outcome_index=1,
        )
        summary = _summarize_scored_dags(scored, random_seed=42, selection_mode="single_extension")

        self.assertEqual(summary["n_extensions"], 1)
        self.assertEqual(summary["selection_mode"], "single_extension")
        self.assertEqual(summary["psi_total_selected"], summary["psi_total_mean"])
        self.assertEqual(summary["psi_total_selected"], summary["psi_total_max"])
        self.assertEqual(summary["psi_total_selected"], summary["psi_total_random"])

    def test_phase2_scoring_prefers_lower_proxy_parent_penalty(self):
        node_names = ["S", "B", "Y"]
        data = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
            ]
        )
        proxy_load = _compute_proxy_load(data, node_names, [0])

        scored = _score_dags_by_phase2(
            [[("S", "Y")], [("B", "Y")]],
            data=data,
            node_names=node_names,
            proxy_load=proxy_load,
            outcome_index=2,
        )
        summary = _summarize_scored_dags(scored, random_seed=42, selection_mode="exact")

        self.assertEqual(summary["selected"].dag_edges, [("B", "Y")])
        self.assertLess(summary["psi_total_selected"], scored[0].psi_total)

    def test_tie_breaking_uses_enumerator_order(self):
        scored = [
            ScoredDag(ordinal=0, dag_edges=[("A", "Y")], psi_total=0.5),
            ScoredDag(ordinal=1, dag_edges=[("B", "Y")], psi_total=0.5),
        ]

        summary = _summarize_scored_dags(scored, random_seed=42, selection_mode="exact")

        self.assertEqual(summary["selected"].dag_edges, [("A", "Y")])

    def test_random_audit_dag_is_seeded(self):
        scored = [
            ScoredDag(ordinal=i, dag_edges=[(f"X{i}", "Y")], psi_total=float(i))
            for i in range(5)
        ]

        first = _summarize_scored_dags(scored, random_seed=7, selection_mode="exact")
        second = _summarize_scored_dags(scored, random_seed=7, selection_mode="exact")

        self.assertEqual(first["random"].dag_edges, second["random"].dag_edges)
        self.assertEqual(first["psi_total_random"], second["psi_total_random"])

    def test_table_display_label_registered(self):
        self.assertEqual(EXPERIMENT_DISPLAY["ges_phase2_only"], "GESPhase2Only")


if __name__ == "__main__":
    unittest.main()
