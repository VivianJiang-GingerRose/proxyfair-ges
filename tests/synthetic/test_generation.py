import numpy as np
import unittest

from src.faircausal.data.synthetic.archetype_definitions import ARCHETYPES
from src.faircausal.data.synthetic.generator import generate_synthetic_dataset


class TestSyntheticGeneration(unittest.TestCase):
    def test_archetype_schema(self):
        # Synthetic paper archetypes should be registered.
        self.assertEqual(set(ARCHETYPES.keys()), {1, 3, 4})
        for _, spec in ARCHETYPES.items():
            self.assertIn(spec.outcome_variable, spec.all_variables)
            self.assertIn(spec.proxy_variable, spec.all_variables)
            self.assertIn(spec.legit_variable, spec.observed_variables)
            self.assertIn((spec.legit_variable, spec.outcome_variable), spec.edges)
            for lv in spec.latent_variables:
                self.assertNotIn(lv, spec.observed_variables)


    def test_discretization_integrity(self):
        out = generate_synthetic_dataset(archetype_id=1, N=500, phi_target=0.5, n_bins=3, seed=10)
        data = out["data"]
        self.assertTrue(np.issubdtype(data.dtype, np.integer))
        self.assertFalse(np.isnan(data.astype(float)).any())
        self.assertGreaterEqual(data.min(), 0)
        self.assertLessEqual(data.max(), 2)


    def test_seed_reproducibility(self):
        a = generate_synthetic_dataset(archetype_id=1, N=500, phi_target=0.5, n_bins=3, seed=123)
        b = generate_synthetic_dataset(archetype_id=1, N=500, phi_target=0.5, n_bins=3, seed=123)
        self.assertTrue(np.array_equal(a["data"], b["data"]))


    def test_latent_exclusion(self):
        out = generate_synthetic_dataset(
            archetype_id=3,
            N=1000,
            phi_target=0.5,
            n_bins=3,
            seed=42,
            confounding_strength=0.7,
        )
        names = out["variable_names"]
        # Latent U must be excluded; observed Z must be present.
        self.assertNotIn("U", names)
        self.assertIn("Z", names)
        # S and V are still correlated through U.
        s_idx = names.index("S")
        v_idx = names.index("V")
        corr = np.corrcoef(out["data"][:, s_idx], out["data"][:, v_idx])[0, 1]
        self.assertGreater(abs(corr), 0.05)


    def test_true_pse_correctness(self):
        out = generate_synthetic_dataset(archetype_id=1, N=500, phi_target=0.5, n_bins=3, seed=7)
        coeffs = out["true_coefficients"]
        expected = coeffs[("S", "V")] * coeffs[("V", "Y")]
        self.assertLess(abs(out["true_pse"]["indirect_proxy"] - expected), 1e-10)


    def test_legit_variable_independence(self):
        """Z must be (near) independent of S and V in archetype 1 and 3."""
        for archetype_id in [1, 3]:
            kwargs = {"archetype_id": archetype_id, "N": 3000, "phi_target": 0.5, "n_bins": 3, "seed": 99}
            if archetype_id == 3:
                kwargs["confounding_strength"] = 0.7
            out = generate_synthetic_dataset(**kwargs)
            names = out["variable_names"]
            z_idx = names.index("Z")
            s_idx = names.index("S")
            v_idx = names.index("V")
            data = out["data"].astype(float)
            corr_zs = abs(np.corrcoef(data[:, z_idx], data[:, s_idx])[0, 1])
            corr_zv = abs(np.corrcoef(data[:, z_idx], data[:, v_idx])[0, 1])
            self.assertLess(corr_zs, 0.15, f"archetype {archetype_id}: |corr(Z,S)|={corr_zs:.3f} too high")
            self.assertLess(corr_zv, 0.15, f"archetype {archetype_id}: |corr(Z,V)|={corr_zv:.3f} too high")


    def test_archetype1_variable_names(self):
        out = generate_synthetic_dataset(archetype_id=1, N=200, phi_target=0.5, n_bins=3, seed=5)
        self.assertEqual(set(out["variable_names"]), {"S", "V", "Z", "Y"})
        self.assertIsNotNone(out["legit_index"])


    def test_archetype3_variable_names(self):
        out = generate_synthetic_dataset(archetype_id=3, N=200, phi_target=0.5, n_bins=3, seed=5, confounding_strength=0.5)
        self.assertEqual(set(out["variable_names"]), {"S", "V", "Z", "Y"})
        self.assertIsNotNone(out["legit_index"])


    def test_archetype4_displacement_generation(self):
        out = generate_synthetic_dataset(archetype_id=4, N=5000, phi_target=0.5, n_bins=5, seed=77)
        self.assertEqual(out["variable_names"], ["C", "S1", "S2", "V1", "V2", "Z", "Y"])
        self.assertEqual(
            set(out["true_edges"]),
            {("C", "S1"), ("C", "S2"), ("S1", "V1"), ("S2", "V2"), ("V1", "Y"), ("V2", "Y"), ("Z", "Y")},
        )
        self.assertEqual(out["protected_indices"], [1, 2])
        self.assertEqual(out["proxy_indices"], [3, 4])
        self.assertEqual(out["legit_indices"], [5])

        cal = out["calibration"]
        self.assertLess(abs(cal["corr_s1_s2_continuous"] - 0.5), 0.05)
        self.assertLess(abs(cal["r2_v1_s1_continuous"] - 0.5), 0.08)
        self.assertLess(abs(cal["r2_v2_s2_continuous"] - 0.5), 0.08)

        names = out["variable_names"]
        z_idx = names.index("Z")
        data_cont = out["continuous_data"]
        for other in ["C", "S1", "S2", "V1", "V2"]:
            corr = abs(np.corrcoef(data_cont[:, z_idx], data_cont[:, names.index(other)])[0, 1])
            self.assertLess(corr, 0.06, f"|corr(Z,{other})|={corr:.3f} too high")


if __name__ == "__main__":
    unittest.main()
