import unittest
from types import SimpleNamespace

import numpy as np

from experiments.evaluate_feature_importance import (
    GROUPS, align_regions, correlation_matrix, forest_decisions, permute_columns, weighted_metrics,
)
from experiments.evaluate_thresholds import REGION_DTYPE
from src.region_features import FEATURE_NAMES


class FeatureImportanceTests(unittest.TestCase):
    def setUp(self):
        self.regions = np.array([("first", 30, 10, 2, 8, 0.6, True),
                                 ("first", 4, 2, 2, 0, 0.4, False)], dtype=REGION_DTYPE)

    def test_alignment_and_feature_order(self):
        self.assertEqual(tuple(name for group in GROUPS.values() for name in group), FEATURE_NAMES)
        aligned = align_regions(self.regions, np.array([4, 30]), np.array([2, 10]), np.array([2, 2]))
        np.testing.assert_array_equal(aligned["id"], [4, 30])
        with self.assertRaises(ValueError):
            align_regions(self.regions, np.array([4, 30]), np.array([2, 11]), np.array([2, 2]))

    def test_permutation_preserves_input_and_group_relationship(self):
        features = np.column_stack((np.arange(15), np.arange(15) + 100, np.arange(15) + 200))
        original = features.copy()
        individual = permute_columns(features, (1,), 42)
        np.testing.assert_array_equal(individual[:, [0, 2]], original[:, [0, 2]])
        np.testing.assert_array_equal(np.sort(individual[:, 1]), original[:, 1])
        joint = permute_columns(features, (0, 2), 43)
        np.testing.assert_array_equal(joint[:, 2] - joint[:, 0], 200)
        np.testing.assert_array_equal(joint[:, 1], original[:, 1])
        np.testing.assert_array_equal(joint, permute_columns(features, (0, 2), 43))
        np.testing.assert_array_equal(features, original)

    def test_weighted_pixels_and_individual_totals(self):
        selected = np.array([True, False])
        counts = weighted_metrics(self.regions, selected)
        totals = {"TP": 2, "FP": 8, "TN": 0, "FN": 2}
        self.assertEqual({k: counts[k] for k in totals}, totals)
        self.assertEqual(counts["IoU"], 2 / 12)
        pieces = [weighted_metrics(self.regions[i:i+1], selected[i:i+1]) for i in range(2)]
        self.assertEqual({key: sum(row[key] for row in pieces) for key in totals}, totals)
        self.assertEqual(weighted_metrics(self.regions[:0], selected[:0])["IoU"], 0)
        self.assertEqual(weighted_metrics(self.regions, np.zeros(2, dtype=bool))["Precision"], 0)

    def test_constant_correlations_and_class_order(self):
        features = np.array([[1, 2, 9], [2, 4, 9], [3, 6, 9]])
        for spearman in (False, True):
            correlation = correlation_matrix(features, spearman=spearman)
            self.assertAlmostEqual(correlation[0, 1], 1)
            self.assertTrue(np.isnan(correlation[2]).all())
        selected, probabilities = forest_decisions(SimpleNamespace(classes_=np.array([1, 0])),
                                                    np.array([[0.5, 0.5], [0.1, 0.9]]))
        np.testing.assert_array_equal(selected, [True, False])
        np.testing.assert_array_equal(probabilities, [0.5, 0.1])
