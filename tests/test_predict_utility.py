import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graph_memory.analysis import predict_utility


class TestPredictUtility(unittest.TestCase):
    def test_pearson_and_spearman_exact(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys_pos = [2.0, 4.0, 6.0, 8.0, 10.0]
        ys_neg = [10.0, 8.0, 6.0, 4.0, 2.0]

        self.assertAlmostEqual(predict_utility.pearson_r(xs, ys_pos), 1.0, places=6)
        self.assertAlmostEqual(predict_utility.pearson_r(xs, ys_neg), -1.0, places=6)
        self.assertAlmostEqual(predict_utility.spearman_rho(xs, ys_pos), 1.0, places=6)
        self.assertAlmostEqual(predict_utility.spearman_rho(xs, ys_neg), -1.0, places=6)

    def test_fit_ols_linear_signal(self):
        # Construct Y = -2 * X1 + 1 * X2 with high R^2
        x1 = [float(i) for i in range(20)]
        x2 = [float((i * 3) % 7) for i in range(20)]
        y = [-2.0 * a + 1.0 * b for a, b in zip(x1, x2)]
        res = predict_utility.fit_ols({"degree_gini": x1, "largest_component_ratio": x2}, y, n_perm=200)
        self.assertGreater(res["r2"], 0.99)
        self.assertLess(res["betas"]["degree_gini"], 0.0)
        self.assertGreater(res["betas"]["largest_component_ratio"], 0.0)
        self.assertLess(res["p_value"], 0.05)

    def test_collect_and_analyze_observations(self):
        from graph_memory.analysis import make_tables
        runs = make_tables.load_runs()
        obs = predict_utility.collect_sample_observations(runs)
        self.assertGreaterEqual(len(obs), 20)
        res = predict_utility.analyze_observations(obs)
        self.assertIn("ols", res)
        self.assertIn("evidence_coverage", res["ols"])


if __name__ == "__main__":
    unittest.main()
