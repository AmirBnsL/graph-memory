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

        self.assertEqual(len(obs_nat), 160)
        self.assertGreater(len(obs_all), len(obs_nat))

        nat_modes = {o["mode"] for o in obs_nat}
        self.assertEqual(nat_modes, {"memory_hipporag", "memory_temporal_bteg", "memory_hipporag2", "memory_lightrag"})
        self.assertNotIn("memory_hipporag_rewired", nat_modes)
        self.assertNotIn("memory_hipporag_flattened", nat_modes)

    def test_h2_falsification_criteria_on_natural_graphs(self):
        from graph_memory.analysis import make_tables
        runs = make_tables.load_runs()
        obs_nat = predict_utility.collect_sample_observations(runs, natural_only=True)
        res = predict_utility.analyze_observations(obs_nat)

        self.assertEqual(res["n"], 160)

        # Baseline subset (Concept Graph + Temporal BTEG, n=80) meets R^2 >= 0.20
        obs_baseline = [o for o in obs_nat if o["mode"] in ("memory_hipporag", "memory_temporal_bteg")]
        res_baseline = predict_utility.analyze_observations(obs_baseline)
        self.assertEqual(res_baseline["n"], 80)
        self.assertGreaterEqual(res_baseline["ols"]["evidence_coverage"]["r2"], 0.20)
        self.assertLess(res_baseline["ols"]["evidence_coverage"]["p_value"], 0.05)
        self.assertGreaterEqual(res_baseline["ols"]["precision_at_k"]["r2"], 0.20)
        self.assertLess(res_baseline["ols"]["precision_at_k"]["p_value"], 0.05)

        # Full expanded natural corpus (n=160 across 4 architectures)
        cov_ols = res["ols"]["evidence_coverage"]
        self.assertGreaterEqual(cov_ols["r2"], 0.10)
        self.assertLess(cov_ols["p_value"], 0.05)

        prec_ols = res["ols"]["precision_at_k"]
        self.assertGreaterEqual(prec_ols["r2"], 0.10)
        self.assertLess(prec_ols["p_value"], 0.05)

        # Max degree ratio negatively predicts retrieval evidence coverage and precision@k
        max_deg_cov = res["correlations"]["max_degree_ratio"]["evidence_coverage"]
        self.assertLess(max_deg_cov["spearman_rho"], 0.0)
        self.assertLess(max_deg_cov["p_value"], 0.05)

        max_deg_prec = res["correlations"]["max_degree_ratio"]["precision_at_k"]
        self.assertLess(max_deg_prec["spearman_rho"], 0.0)
        self.assertLess(max_deg_prec["p_value"], 0.05)

        # Edge density positively predicts retrieval evidence coverage
        dens_cov = res["correlations"]["edge_density"]["evidence_coverage"]
        self.assertGreater(dens_cov["spearman_rho"], 0.25)
        self.assertLess(dens_cov["p_value"], 0.05)

    def test_table1_schema_filtering(self):
        from graph_memory.analysis import make_tables
        runs = make_tables.load_runs()
        calibrated_runs = [d for d in runs if d.get("graph_metrics") and d.get("graph_metrics_by_sample")]

        # All calibrated runs have complete per-sample breakdowns and 25 samples
        self.assertEqual(len(calibrated_runs), 12)
        for d in calibrated_runs:
            self.assertEqual(len(d["graph_metrics_by_sample"]), 25)
            # Flattened has 0 density, all non-flattened have positive density
            if "flattened" not in d["mode"]:
                self.assertGreater(d["graph_metrics"]["edge_density"], 0.0)

    def test_regime_breakdown_and_table4_rendering(self):
        import io
        import sys
        from graph_memory.analysis import make_tables

        runs = make_tables.load_runs()
        obs_nat = predict_utility.collect_sample_observations(runs, natural_only=True)
        regimes = predict_utility.compute_regime_splits(obs_nat)

        self.assertIn("hub_concentration", regimes)
        self.assertIn("super_hub_dominance", regimes)
        self.assertIn("component_connectivity", regimes)

        # Super-Hub Dominance regime (Max Degree Ratio) has massive, highly significant degradation
        hub = regimes["super_hub_dominance"]
        self.assertGreater(hub["targets"]["evidence_coverage"]["delta"], 0.15)
        self.assertLess(hub["targets"]["evidence_coverage"]["p_value"], 0.05)
        self.assertGreater(hub["targets"]["delta_f1_vs_flat"]["delta"], 0.15)
        self.assertLess(hub["targets"]["delta_f1_vs_flat"]["p_value"], 0.05)
        self.assertGreater(hub["targets"]["precision_at_k"]["delta"], 0.04)
        self.assertLess(hub["targets"]["precision_at_k"]["p_value"], 0.05)

        # Baseline subset (n=80) verifies component connectivity contrast
        obs_baseline = [o for o in obs_nat if o["mode"] in ("memory_hipporag", "memory_temporal_bteg")]
        base_regimes = predict_utility.compute_regime_splits(obs_baseline)
        conn_base = base_regimes["component_connectivity"]
        self.assertGreater(conn_base["targets"]["evidence_coverage"]["delta"], 0.25)
        self.assertLess(conn_base["targets"]["evidence_coverage"]["p_value"], 0.05)

        # Output capture check
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            predict_utility.print_h2_diagnostics(runs)
        finally:
            sys.stdout = old_stdout

        out = buf.getvalue()
        self.assertIn("### Hypothesis 2 (Table 4): Topological Regime Breakdown", out)
        self.assertIn("Hub Concentration", out)
        self.assertIn("Super-Hub Dominance", out)
        self.assertIn("Component Connectivity", out)




if __name__ == "__main__":
    unittest.main()
