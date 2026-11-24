import unittest

import numpy as np

from src.region_labeling import label_regions


class RegionLabelingTests(unittest.TestCase):
    def test_sparse_ids_and_inclusive_thresholds(self):
        regions = np.repeat([[90], [2], [700]], 10, axis=1)
        label = np.where(np.arange(10) < np.array([[7], [1], [3]]), 5, 2)
        result = label_regions(regions, label)
        np.testing.assert_array_equal(result.region_ids, [2, 90, 700])
        np.testing.assert_array_equal(result.pixel_counts, [10, 10, 10])
        np.testing.assert_array_equal(result.target_counts, [1, 7, 3])
        np.testing.assert_array_equal(result.labels, [0, 1, -1])
        np.testing.assert_array_equal(result.region_ids[result.pixel_indices], regions)

    def test_all_regions_are_negative_when_stain_is_absent(self):
        result = label_regions(np.array([[0, 75]]), np.array([[0, 2]]))
        np.testing.assert_array_equal(result.labels, [0, 0])
        np.testing.assert_array_equal(result.target_counts, [0, 0])
