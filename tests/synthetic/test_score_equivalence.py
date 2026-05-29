import unittest

import numpy as np

from src.faircausal.core.fairness_scoring import (
    compute_edge_proxy_fractions,
    compute_outcome_relevance,
    compute_symmetric_penalty_from_pi,
)


def _total_symmetric_penalty(parents_by_child, psi_sym):
    """Compute total Phase-1 symmetric penalty from parent sets."""
    total = 0.0
    for child, parents in parents_by_child.items():
        for parent in parents:
            total += float(psi_sym[(child, parent)])
    return total


class TestScoreEquivalence(unittest.TestCase):
    def test_symmetric_phase1_penalty_is_orientation_invariant_within_mec(self):
        # Three-node chain MEC: 0-1-2 with no v-structures.
        # DAG A: 0 -> 1 -> 2
        # DAG B: 2 -> 1 -> 0
        # Same skeleton, same v-structure pattern => Markov equivalent.
        rng = np.random.default_rng(7)
        data = rng.integers(low=0, high=4, size=(400, 3), endpoint=False)

        protected_attr_indices = [0]
        outcome_index = 2

        pi_sigma = compute_edge_proxy_fractions(
            data=data,
            protected_attr_indices=protected_attr_indices,
            tau_c=5.0,
            min_group_size=10,
        )
        rho_adj = compute_outcome_relevance(data=data, outcome_index=outcome_index)
        psi_sym = compute_symmetric_penalty_from_pi(pi_sigma=pi_sigma, rho_adj=rho_adj)

        dag_a_parents = {
            0: [],
            1: [0],
            2: [1],
        }
        dag_b_parents = {
            0: [1],
            1: [2],
            2: [],
        }

        total_a = _total_symmetric_penalty(dag_a_parents, psi_sym)
        total_b = _total_symmetric_penalty(dag_b_parents, psi_sym)

        self.assertAlmostEqual(total_a, total_b, places=10)


if __name__ == "__main__":
    unittest.main()
