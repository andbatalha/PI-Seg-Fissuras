import unittest
from fractions import Fraction
from itertools import product

import numpy as np

from experiments.evaluate_slic_oracle import (
    ImageRegions, evaluate_selection, majority_selection, max_iou_selection, summarize_occupancy,
)


class SlicOracleTests(unittest.TestCase):
    def test_majority_and_empty_stain(self):
        areas = np.array([10, 10, 10, 10, 10])
        stain = np.array([0, 10, 5, 6, 1])
        np.testing.assert_array_equal(majority_selection(areas, stain), [0, 1, 0, 1, 0])
        summary = summarize_occupancy(areas, stain)
        self.assertEqual((summary["without_stain"], summary["fully_stain"], summary["partially_stain"]),
                         (1, 1, 3))
        self.assertEqual([item["regions"] for item in summary["bins"]], [1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        selected, cutoff = max_iou_selection(areas, np.zeros_like(stain))
        self.assertFalse(selected.any())
        self.assertIsNone(cutoff)

    def test_tied_iou_prefers_shorter_prefix(self):
        selected, cutoff = max_iou_selection(np.array([1, 3]), np.array([1, 1]))
        np.testing.assert_array_equal(selected, [1, 0])
        self.assertEqual(cutoff, 1.0)

    def test_global_optimum_against_exhaustive_search(self):
        rng = np.random.default_rng(42)
        for _ in range(150):
            areas = rng.integers(1, 15, size=7)
            stain = rng.integers(0, areas + 1)
            total = int(stain.sum())

            def iou(selection):
                union = total + int((areas - stain) @ selection)
                return Fraction(int(stain @ selection), union) if union else Fraction(0)

            optimized, _ = max_iou_selection(areas, stain)
            brute_force = max(iou(selection) for selection in product((0, 1), repeat=areas.size))
            self.assertEqual(iou(optimized), brute_force)
            self.assertGreaterEqual(iou(optimized), iou(majority_selection(areas, stain)))

    def test_global_decisions_reconstruct_sparse_ids_and_aggregate_counts(self):
        positive = ImageRegions("positive", np.array([[80, 80], [3, 3]]), np.array([3, 80]),
                                np.array([[1, 0], [1, 1]], dtype=np.uint8),
                                np.array([2, 2]), np.array([2, 1]), 0)
        negative = ImageRegions("negative", np.array([[71, 71]]), np.array([71]),
                                np.array([[0, 0]], dtype=np.uint8), np.array([2]), np.array([0]), 0)
        values = np.concatenate((positive.areas, negative.areas))
        counts = np.concatenate((positive.stain_pixels, negative.stain_pixels))
        selected, cutoff = max_iou_selection(values, counts)
        report, mask = evaluate_selection(positive, selected[:2])
        empty, empty_mask = evaluate_selection(negative, selected[2:])
        self.assertEqual(cutoff, 0.5)
        np.testing.assert_array_equal(mask, [[1, 1], [1, 1]])
        np.testing.assert_array_equal(empty_mask, [[0, 0]])
        self.assertEqual(empty["metrics"]["IoU"], 0)
        totals = {key: report["counts"][key] + empty["counts"][key] for key in report["counts"]}
        self.assertEqual(totals, {"TP": 3, "TN": 2, "FP": 1, "FN": 0})
