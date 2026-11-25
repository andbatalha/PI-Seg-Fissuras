import unittest
from unittest.mock import patch

import numpy as np
from skimage.color import rgb2lab

from src.feature_extractor import FEATURE_GROUPS, extract_features
from src.multiscale_context import MULTISCALE_NAMES, extract_multiscale_context
from src.region_features import FEATURE_NAMES, extract_region_features


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.rgb = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
        self.regions = np.array([[300, 9, 62, 24], [8, 75, 115, 27], [55, 20, 20, 150]])
        self.regions = self.regions.repeat(2, axis=0).repeat(2, axis=1)
        self.ids, self.base = extract_region_features(self.rgb, self.regions)
        graph_ids, multiscale = extract_multiscale_context(self.rgb, self.regions)
        np.testing.assert_array_equal(self.ids, graph_ids)
        self.full = np.column_stack((self.base, multiscale))

    def test_every_group_can_be_selected_individually(self):
        names = (*FEATURE_NAMES, *MULTISCALE_NAMES)
        for group, columns in FEATURE_GROUPS.items():
            with self.subTest(group=group):
                ids, values, actual_names = extract_features(self.rgb, self.regions, [group])
                np.testing.assert_array_equal(ids, self.ids)
                self.assertEqual(actual_names, columns)
                np.testing.assert_array_equal(values, self.full[:, [names.index(name) for name in columns]])
                self.assertTrue(np.isfinite(values).all())

    def test_baseline_and_experimental_configurations_preserve_order_and_values(self):
        groups = tuple(FEATURE_GROUPS)
        names = (*FEATURE_NAMES, *MULTISCALE_NAMES)
        for group_count, column_count in ((6, 15), (7, 21), (8, 25)):
            with self.subTest(columns=column_count):
                ids, matrix, actual_names = extract_features(self.rgb, self.regions, reversed(groups[:group_count]))
                np.testing.assert_array_equal(ids, self.ids)
                np.testing.assert_array_equal(matrix, self.full[:, :column_count])
                self.assertEqual(actual_names, names[:column_count])

    def test_combination_skips_glcm_and_robust_percentiles_and_shares_lab(self):
        with (
            patch("src.region_features._masked_glcm_features", side_effect=AssertionError("GLCM desnecessária")),
            patch("src.multiscale_context.np.percentile", side_effect=AssertionError("Percentil desnecessário")),
            patch("src.feature_extractor.rgb2lab", wraps=rgb2lab) as conversion,
        ):
            ids, matrix, names = extract_features(
                self.rgb, self.regions, ["multiscale_lab", "entropy", "lab_std", "entropy"],
            )
        conversion.assert_called_once_with(self.rgb)
        self.assertEqual(names, (*FEATURE_NAMES[3:6], "hist_entropy", *MULTISCALE_NAMES[:6]))
        np.testing.assert_array_equal(ids, self.ids)
        np.testing.assert_array_equal(matrix, self.full[:, [3, 4, 5, 9, *range(15, 21)]])

    def test_empty_and_unknown_selection(self):
        with self.assertRaisesRegex(ValueError, "pelo menos um"):
            extract_features(self.rgb, self.regions, [])
        with self.assertRaisesRegex(ValueError, "grupo_desconhecido"):
            extract_features(self.rgb, self.regions, ["gradient", "grupo_desconhecido"])
