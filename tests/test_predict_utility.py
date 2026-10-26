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

    def test_natural_substrate_filtering(self):
        from graph_memory.analysis import make_tables
        runs = make_tables.load_runs()
        obs_all = predict_utility.collect_sample_observations(runs, natural_only=False)
        obs_nat = predict_utility.collect_sample_observations(runs, natural_only=True)

        self.assertEqual(len(obs_nat), 80)
        self.assertGreater(len(obs_all), len(obs_nat))

        nat_modes = {o["mode"] for o in obs_nat}
        self.assertEqual(nat_modes, {"memory_hipporag", "memory_temporal_bteg"})
        self.assertNotIn("memory_hipporag_rewired", nat_modes)
        self.assertNotIn("memory_hipporag_flattened", nat_modes)

    def test_h2_falsification_criteria_on_natural_graphs(self):
        from graph_memory.analysis import make_tables
        runs = make_tables.load_runs()
        obs_nat = predict_utility.collect_sample_observations(runs, natural_only=True)
        res = predict_utility.analyze_observations(obs_nat)

        self.assertEqual(res["n"], 80)

        # OLS on retrieval evidence coverage >= 0.20 and p < 0.05
        cov_ols = res["ols"]["evidence_coverage"]
        self.assertGreaterEqual(cov_ols["r2"], 0.20)
        self.assertLess(cov_ols["p_value"], 0.05)

        # OLS on retrieval precision@k >= 0.20 and p < 0.05
        prec_ols = res["ols"]["precision_at_k"]
        self.assertGreaterEqual(prec_ols["r2"], 0.20)
        self.assertLess(prec_ols["p_value"], 0.05)

        # Rank correlations: Degree Gini negatively predicts delta F1 vs Flat
        gini_df1 = res["correlations"]["degree_gini"]["delta_f1_vs_flat"]
        self.assertLessEqual(gini_df1["spearman_rho"], -0.30)
        self.assertLess(gini_df1["p_value"], 0.05)

        # Max degree ratio negatively predicts retrieval evidence coverage
        max_deg_cov = res["correlations"]["max_degree_ratio"]["evidence_coverage"]
        self.assertLessEqual(max_deg_cov["spearman_rho"], -0.30)
        self.assertLess(max_deg_cov["p_value"], 0.05)

        # Largest component ratio positively predicts delta F1 vs Flat
        lcc_df1 = res["correlations"]["largest_component_ratio"]["delta_f1_vs_flat"]
        self.assertGreaterEqual(lcc_df1["spearman_rho"], 0.25)
        self.assertLess(lcc_df1["p_value"], 0.05)

    def test_table1_schema_filtering(self):
        from graph_memory.analysis import make_tables
        runs = make_tables.load_runs()
        calibrated_runs = [d for d in runs if d.get("graph_metrics") and d.get("graph_metrics_by_sample")]

        # All calibrated runs have complete per-sample breakdowns and 25 samples
        self.assertEqual(len(calibrated_runs), 8)
        for d in calibrated_runs:
            self.assertEqual(len(d["graph_metrics_by_sample"]), 25)
            # Flattened has 0 density, all non-flattened have positive density
            if "flattened" not in d["mode"]:
                self.assertGreater(d["graph_metrics"]["edge_density"], 0.0)



if __name__ == "__main__":
    unittest.main()
