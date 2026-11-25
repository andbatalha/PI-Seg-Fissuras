import unittest
from types import SimpleNamespace

import numpy as np

from experiments.evaluate_false_positives import (
    OCCUPANCY_BINS, PROBABILITY_BINS, class_pixel_counts, class_rows,
    occupancy_bin, probability_bin, region_rows, summarize_presence, summarize_regions,
)
from src.region_labeling import label_regions


class FalsePositiveTests(unittest.TestCase):
    def test_original_classes_and_global_sum(self):
        label = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        mask = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8)
        totals, predicted = class_pixel_counts(label, mask)
        rows = class_rows("GLOBAL", totals, predicted)
        self.assertEqual([r["false_positives"] for r in rows], [1, 0, 1, 1, 0])
        first, second = (class_pixel_counts(l, m) for l, m in zip(np.split(label, 2), np.split(mask, 2)))
        np.testing.assert_array_equal(first[0] + second[0], totals)
        np.testing.assert_array_equal(first[1] + second[1], predicted)
        self.assertEqual(class_rows("empty", np.zeros(6), np.zeros(6))[0]["false_positive_rate"], 0)

    def test_exact_bin_edges(self):
        cases = {0: 0, 1: 1, 10: 1, 11: 2, 20: 2, 30: 3, 40: 4, 50: 5,
                 60: 6, 61: 7, 69: 7, 70: 8, 80: 8, 81: 9, 90: 9, 91: 10, 100: 11}
        for count, index in cases.items():
            self.assertEqual(occupancy_bin(count, 100), OCCUPANCY_BINS[index])
        for index, value in enumerate((0.50, 0.60, 0.70, 0.80, 0.90)):
            self.assertEqual(probability_bin(value), PROBABILITY_BINS[index])
            if index:
                self.assertEqual(probability_bin(np.nextafter(value, 0)), PROBABILITY_BINS[index - 1])
        self.assertEqual(probability_bin(1), PROBABILITY_BINS[-1])

    def test_sparse_id_join_preserves_original_decision_and_checks_d01(self):
        labeling = label_regions(np.array([[800, 4, 4]]), np.array([[0, 5, 0]]))
        prediction = SimpleNamespace(region_ids=np.array([4, 800]),
                                     predictions=np.array([0, 1]), probabilities=np.array([0.5, 1]))
        oracle = {("image", 4): (2, 1, 1, 0.5, 1), ("image", 800): (1, 0, 1, 0.0, 0)}
        rows = region_rows("image", prediction, labeling, oracle.copy())
        self.assertEqual([row["FP"] for row in rows], [0, 1])
        self.assertEqual([row["oracle_max_iou_prediction"] for row in rows], [1, 0])
        self.assertEqual(summarize_regions(rows, 1)["predicted_stain_regions"], 1)
        oracle[("image", 800)] = (2, 0, 2, 0.0, 0)
        with self.assertRaises(ValueError):
            region_rows("image", prediction, labeling, oracle)

    def test_presence_without_stain_and_empty_group(self):
        group = [{"TP": 0, "TN": 4, "FP": 1, "FN": 0}, {"TP": 0, "TN": 5, "FP": 0, "FN": 0}]
        report = summarize_presence(group, 1)
        self.assertEqual(report["images_with_fp_share"], 0.5)
        self.assertEqual(report["false_positive_rate"], 0.1)
        self.assertEqual(report["median_fp_per_image"], 0.5)
        self.assertEqual(summarize_presence([], 0)["false_positive_rate"], 0)
        self.assertIsNone(summarize_regions([], 0)["mean_probability"])
