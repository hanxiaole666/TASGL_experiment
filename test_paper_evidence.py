#!/usr/bin/env python3
"""paper_evidence.py 的统计、完整性与真实证据确定性回归。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import paper_evidence as pe


REAL_EVIDENCE = os.environ.get("TASGL_EVIDENCE_ZIP")
REAL_V3 = os.environ.get("TASGL_V3_ZIP")
REAL_V4 = os.environ.get("TASGL_V4_ZIP")


class TestPaperStatistics(unittest.TestCase):
    def test_sample_and_population_standard_deviation(self):
        mean, sample_sd = pe.mean_sd([1, 2, 3], ddof=1)
        self.assertEqual(mean, 2.0)
        self.assertEqual(sample_sd, 1.0)
        _, population_sd = pe.mean_sd([1, 2, 3], ddof=0)
        self.assertAlmostEqual(population_sd, math.sqrt(2 / 3), places=12)
        with self.assertRaises(ValueError):
            pe.mean_sd([1], ddof=1)

    def test_exact_wilcoxon_five_pair_minimum(self):
        result = pe.exact_wilcoxon_signed_rank([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["p_value"], 0.0625)
        self.assertEqual(result["n_nonzero"], 5)
        self.assertEqual(result["minimum_exact_two_sided_p"], 0.0625)

    def test_selective_full_coverage_must_equal_one_minus_f1(self):
        pe.check_selective_curve(0.25, [0.5, 1.0], [0.8, 0.75], "synthetic")
        with self.assertRaisesRegex(ValueError, "1-F1"):
            pe.check_selective_curve(0.25, [0.5, 1.0], [0.8, 0.70], "synthetic")

    def test_hash_mismatch_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence.zip"
            with ZipFile(evidence, "w") as archive:
                archive.writestr("placeholder.json", "{}")
            out = root / "out"
            args = pe.parse_args(["--evidence_zip", str(evidence),
                                  "--expect_sha256", "0" * 64,
                                  "--output_dir", str(out)])
            with self.assertRaisesRegex(ValueError, "SHA-256 不匹配"):
                pe.generate(args)
            self.assertFalse(out.exists())

    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "occupied"
            root.mkdir()
            (root / "keep.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "禁止覆盖"):
                pe.OutputWriter(root)
            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"),
                             "do not overwrite")


@unittest.skipUnless(REAL_EVIDENCE, "设置 TASGL_EVIDENCE_ZIP 后运行真实证据集成回归")
class TestRealEvidenceReproduction(unittest.TestCase):
    def test_real_evidence_is_deterministic_and_matches_locked_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = []
            for name in ("run_a", "run_b"):
                argv = ["--evidence_zip", REAL_EVIDENCE, "--output_dir", str(root / name)]
                if REAL_V3:
                    argv += ["--v3_zip", REAL_V3]
                if REAL_V4:
                    argv += ["--v4_zip", REAL_V4]
                args = pe.parse_args(argv)
                result = pe.generate(args)
                self.assertFalse(result["protocol5_all_pass"])
                self.assertIsNone(result["protocol6_winner"])
                self.assertFalse(result["confirmation_authorized"])
                outputs.append(root / name)
            files_a = {str(p.relative_to(outputs[0])): p.read_bytes()
                       for p in outputs[0].rglob("*") if p.is_file()}
            files_b = {str(p.relative_to(outputs[1])): p.read_bytes()
                       for p in outputs[1].rglob("*") if p.is_file()}
            self.assertEqual(files_a, files_b)

            manifest = json.loads(files_a["manifest.json"])
            self.assertEqual(hashlib.sha256(files_a["manifest.json"]).hexdigest(),
                             files_a["manifest.sha256"].decode().strip())
            for relative, expected in manifest["outputs"].items():
                self.assertEqual(hashlib.sha256(files_a[relative]).hexdigest(), expected,
                                 relative)
            self.assertTrue(json.loads(files_a["validation_report.json"])["valid"])

            main = json.loads(files_a["tables/table_02_main_results.json"])
            tasgl = next(r for r in main if r["model_key"] == "tasgl")
            self.assertAlmostEqual(tasgl["f1_mean"], 0.5038113415948291, places=12)
            self.assertAlmostEqual(tasgl["f1_sample_sd"], 0.0861656648046072, places=12)
            thresholds = [r["threshold"] for r in
                          json.loads(files_a["tables/table_02_per_seed_operating_points.json"])
                          if r["model_key"] == "tasgl"]
            self.assertEqual(thresholds, [0.54, 0.9400000000000001, 0.88,
                                          0.9400000000000001, 0.5])

            robustness = json.loads(files_a["tables/table_06_robustness.json"])
            clean = next(r for r in robustness if r["model_key"] == "tasgl" and
                         r["condition"] == "clean")
            self.assertEqual(clean["ddof"], 0)
            clean_values = [0.4017910447761194, 0.6267573696145124]
            clean_mean = sum(clean_values) / 2
            clean_population_sd = math.sqrt(
                sum((v - clean_mean) ** 2 for v in clean_values) / 2)
            self.assertAlmostEqual(clean["mean"], clean_mean, places=12)
            self.assertAlmostEqual(clean["sd"], clean_population_sd, places=12)

            curves = json.loads(files_a["tables/table_07a_selective_risk.json"])
            e7 = json.loads(files_a["tables/table_07a_uncertainty_e7.json"])
            for key in ("tasgl", "edl"):
                full = next(r for r in curves if r["model_key"] == key and r["full_coverage"])
                f1 = next(r["f1"] for r in e7 if r["model_key"] == key)
                self.assertAlmostEqual(full["selective_risk_1_minus_f1"], 1 - f1, places=12)

            p6 = json.loads(files_a["derived_metrics.json"])["protocol6"]
            self.assertEqual(p6["run_count"], 28)
            self.assertTrue(all(not r["eligible"] for r in p6["rows"]))
            for relative, payload in files_a.items():
                if relative.endswith(".tex"):
                    for line in payload.decode("utf-8").splitlines():
                        terminal = len(line) - len(line.rstrip("\\"))
                        self.assertNotEqual(terminal, 1,
                                            f"{relative}: LaTeX 行尾只有一个反斜杠")


if __name__ == "__main__":
    unittest.main(verbosity=2)
