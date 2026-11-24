import unittest

import numpy as np
from skimage.color import rgb2lab

from src.region_features import FEATURE_GROUPS, FEATURE_NAMES, extract_region_features


class RegionFeaturesTests(unittest.TestCase):
    def test_uniform_region_and_feature_order(self):
        rgb = np.full((5, 6, 3), [30, 140, 210], dtype=np.uint8)
        ids, features = extract_region_features(rgb, np.full((5, 6), 900))
        self.assertEqual(FEATURE_NAMES, (
            "mean_l", "mean_a", "mean_b", "std_l", "std_a", "std_b",
            "delta_l", "delta_a", "delta_b", "hist_entropy", "glcm_contrast",
            "glcm_homogeneity", "glcm_asm", "glcm_correlation", "mean_gradient",
        ))
        self.assertEqual(tuple(map(len, FEATURE_GROUPS.values())), (6, 3, 1, 4, 1))
        np.testing.assert_array_equal(ids, [900])
        self.assertEqual(features.shape, (1, 15))
        np.testing.assert_allclose(features[0, :3], rgb2lab(rgb)[0, 0])
        np.testing.assert_allclose(features[0, 3:], [0] * 8 + [1, 1, 1, 0], atol=1e-12)

    def test_sparse_ids_and_signed_neighborhood(self):
        rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        rgb[:, 2:] = 255
        regions = np.tile([700, 700, 3, 3], (3, 1))
        ids, features = extract_region_features(rgb, regions)
        np.testing.assert_array_equal(ids, [3, 700])
        np.testing.assert_allclose(features[:, 6], [100, -100])
        self.assertTrue(np.isfinite(features).all())
