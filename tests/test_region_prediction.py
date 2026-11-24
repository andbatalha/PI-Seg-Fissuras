import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.region_features import FEATURE_NAMES
from src.region_prediction import predict_regions, reconstruct_binary_mask


class RegionPredictionTests(unittest.TestCase):
    def test_reconstruct_by_id_even_when_predictions_are_out_of_order(self):
        for high in (2, 7000):
            with self.subTest(high=high):
                regions = np.array([[1, high], [high, 1]])
                mask = reconstruct_binary_mask(regions, np.array([high, 1]), np.array([0, 1]))
                self.assertEqual(mask.dtype, np.uint8)
                np.testing.assert_array_equal(mask, [[1, 0], [0, 1]])
                with self.assertRaises(ValueError):
                    reconstruct_binary_mask(regions, np.array([1, -1]), np.array([0, 1]))

    def test_prediction_uses_artifact_configuration_and_all_regions(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        regions = np.array([[800, 7], [7, 800]])
        ids = np.array([7, 800])
        features = np.zeros((2, len(FEATURE_NAMES)))
        model = Mock(n_features_in_=len(FEATURE_NAMES))
        model.predict.return_value = np.array([1, 0])
        slic = {"n_segments": 4, "compactness": 12, "sigma": 0, "start_label": 1}
        artifact = {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "pipeline_config": {
                "slic": slic,
                "features": {"neighborhood_radius": 3, "quantization_levels": 32},
            },
        }
        with patch("src.region_prediction.segment_regions", return_value=regions) as segment:
            with patch("src.region_prediction.extract_region_features", return_value=(ids, features)) as extract:
                result = predict_regions(artifact, rgb)
        segment.assert_called_once_with(rgb, **slic)
        extract.assert_called_once_with(rgb, regions, neighborhood_radius=3)
        model.predict.assert_called_once_with(features)
        np.testing.assert_array_equal(result.region_ids, ids)
        np.testing.assert_array_equal(result.mask, [[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            predict_regions({**artifact, "feature_names": FEATURE_NAMES[::-1]}, rgb)
