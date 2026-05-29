import numpy as np
import unittest

from src.faircausal.data.synthetic.calibration import calibrate_sv_coefficient, proxy_strength_monotone_check
from src.faircausal.data.synthetic.generator import generate_synthetic_dataset


class TestSyntheticCalibration(unittest.TestCase):
    def test_proxy_strength_monotone(self):
        coeffs = (0.1, 0.5, 1.0, 2.0)
        phis = proxy_strength_monotone_check(coeffs, n_bins=3, N=5000, seed=0)
        values = [phis[c] for c in coeffs]
        self.assertLessEqual(values[0], values[1])
        self.assertLessEqual(values[1], values[2])
        self.assertLessEqual(values[2], values[3])


    def test_calibration_tolerance(self):
        for phi_target in (0.1, 0.5, 0.9):
            out = generate_synthetic_dataset(
                archetype_id=1,
                N=4000,
                phi_target=phi_target,
                n_bins=3,
                seed=100 + int(phi_target * 100),
            )
            achieved = out["calibration"]["phi_achieved"]
            self.assertTrue(np.isfinite(achieved))
            self.assertLessEqual(abs(achieved - phi_target), 0.05)


    def test_category_diagnostic_flag(self):
        out = generate_synthetic_dataset(archetype_id=1, N=30, phi_target=0.5, n_bins=6, seed=9)
        min_freq = min(v["min_category_freq"] for v in out["diagnostics"]["category"].values())
        self.assertLessEqual(min_freq, 5)


    def test_calibration_returns_positive_coefficient(self):
        coeff = calibrate_sv_coefficient(phi_target=0.5, n_bins=3, N=3000, seed=1)
        self.assertGreater(coeff, 0)


if __name__ == "__main__":
    unittest.main()
