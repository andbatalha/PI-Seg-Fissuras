import unittest
from fractions import Fraction

import numpy as np

from experiments.evaluate_thresholds import (
    REGION_DTYPE, apply_threshold, best_index, choose_optima, flow_counts, summarize_counts,
    threshold_curve,
)


def make_regions(probabilities, areas, stain):
    return np.array([("a" if i % 2 else "b", i + 30, area, positive, area - positive,
                      probability, probability > 0.5)
                     for i, (probability, area, positive) in enumerate(zip(probabilities, areas, stain))],
                    dtype=REGION_DTYPE)


class ThresholdTests(unittest.TestCase):
    def test_thresholds_keep_equal_probabilities_together(self):
        regions = make_regions([0, 0.5, 0.5, 1], [4, 2, 10, 3], [0, 0, 6, 3])
        curve = threshold_curve(regions, np.array([np.inf, 1, 0.5, 0]))
        np.testing.assert_array_equal(curve["positive_regions"], [0, 1, 3, 4])
        np.testing.assert_array_equal(curve["TP"], [0, 3, 9, 9])
        np.testing.assert_array_equal(curve["FP"], [0, 0, 6, 10])
        baseline = flow_counts(regions, regions["original"])
        self.assertEqual(baseline, {"TP": 3, "TN": 10, "FP": 0, "FN": 6})
        self.assertTrue(apply_threshold(regions["probability"], 0).all())
        self.assertFalse(apply_threshold(regions["probability"], None).any())
        self.assertEqual(int(apply_threshold(regions["probability"], 1).sum()), 1)

    def test_exact_cut_matches_exhaustive_evaluation(self):
        rng = np.random.default_rng(42)
        for _ in range(70):
            areas = rng.integers(1, 20, 12)
            regions = make_regions(rng.integers(0, 5, 12) / 4, areas, rng.integers(0, areas + 1))
            cuts = np.r_[np.inf, np.unique(regions["probability"])[::-1]]
            reports = [flow_counts(regions, apply_threshold(regions["probability"], cut)) for cut in cuts]
            expected = max(range(len(cuts)), key=lambda i: Fraction(
                reports[i]["TP"], sum(reports[i][key] for key in ("TP", "FP", "FN")) or 1,
            ))
            selected = choose_optima(threshold_curve(regions, cuts), flow_counts(regions, regions["original"]))
            self.assertEqual(selected["max_iou"], expected)
            self.assertEqual(selected["max_f1"], expected)

    def test_counts_aggregate_and_empty_positive_support(self):
        regions = make_regions([0.25, 0.75], [5, 2], [0, 0])
        selected = apply_threshold(regions["probability"], 0.5)
        counts = flow_counts(regions, selected)
        pieces = [flow_counts(regions[i:i+1], selected[i:i+1]) for i in range(len(regions))]
        self.assertEqual(counts, {key: sum(row[key] for row in pieces) for key in counts})
        self.assertEqual(summarize_counts(counts, 1, 1)["IoU"], 0)
        summary = summarize_counts({"TP": 0, "TN": 0, "FP": 0, "FN": 0}, 0, 0)
        self.assertEqual(summary["Specificity"], 0)
        self.assertEqual(summary["false_positive_rate"], 0)
        curve = threshold_curve(regions, np.array([np.inf, 0.75, 0.25]))
        self.assertEqual(best_index(curve, np.ones(3, dtype=bool)), 0)
        self.assertIsNone(choose_optima(curve, counts)["precision_at_least_30"])
