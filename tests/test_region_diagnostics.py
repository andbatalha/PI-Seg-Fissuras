import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.evaluation import count_binary_pixels
from src.region_features import FEATURE_NAMES
from src.region_labeling import label_regions
from src.region_prediction import predict_regions
from visualization.inspection import apply_threshold, clicked_region, error_overlay, region_counts


class RegionDiagnosticsTests(unittest.TestCase):
    def test_click_coordinates_scale_to_original_region_ids(self):
        regions = np.array([[10, 90], [200, 900]])
        for x, y, region in ((0, 0, 10), (100, 0, 90), (99, 50, 200), (199, 99, 900)):
            self.assertEqual(clicked_region(regions, dict(x=x, y=y, width=200, height=100)), region)
        self.assertIsNone(clicked_region(regions, dict(x=200, y=10, width=200, height=100)))

    def test_probability_column_uses_classes_and_features_passed_to_model(self):
        regions = np.array([[10, 90]])
        features = np.arange(30).reshape(2, 15)
        model = Mock(classes_=np.array([1, 0]), n_features_in_=15)
        model.predict.return_value = np.array([1, 0])
        model.predict_proba.return_value = np.array([[0.6, 0.4], [0.5, 0.5]])
        artifact = {"model": model, "feature_names": FEATURE_NAMES, "pipeline_config": {
            "slic": {}, "features": {"quantization_levels": 32, "neighborhood_radius": 5},
        }}
        with (
            patch("src.region_prediction.segment_regions", return_value=regions),
            patch("src.region_prediction.extract_region_features", return_value=(np.array([10, 90]), features)),
        ):
            result = predict_regions(artifact, np.zeros((1, 2, 3)), include_details=True)
        self.assertIs(result.features, features)
        model.predict.assert_called_once_with(features)
        model.predict_proba.assert_called_once_with(features)
        np.testing.assert_array_equal(result.region_ids, [10, 90])
        np.testing.assert_array_equal(result.probabilities, [0.6, 0.5])

    def test_threshold_occupancy_and_confusion(self):
        regions = np.array([[10, 90, 500], [10, 90, 500]])
        reference = np.array([[0, 1, 1], [1, 1, 1]], dtype=np.uint8)
        labeling = label_regions(regions, reference * 5)
        baseline = np.array([0, 0, 1], dtype=np.uint8)
        data = {"reference": reference, "labeling": labeling, "original_predictions": baseline,
                "prediction": baseline[labeling.pixel_indices], "probabilities": np.array([0, 0.5, 1])}
        np.testing.assert_array_equal(labeling.target_fractions, [0.5, 1, 1])
        self.assertIs(apply_threshold(data, 0.5)[0], data)
        for threshold, expected in ((0, [1, 1, 1]), (0.49, [0, 1, 1]), (1, [0, 0, 1])):
            with self.subTest(threshold=threshold):
                adjusted, predictions = apply_threshold(data, threshold)
                np.testing.assert_array_equal(predictions, expected)
                np.testing.assert_array_equal(adjusted["prediction"], [expected, expected])
                rows = [region_counts(int(area), int(target), int(prediction)) for area, target, prediction
                        in zip(labeling.pixel_counts, labeling.target_counts, predictions)]
                totals = {name: sum(row[name] for row in rows) for name in rows[0]}
                self.assertEqual(totals, adjusted["counts"])
                self.assertEqual(totals, count_binary_pixels(reference, adjusted["prediction"]))

    def test_error_filters_only_change_rendered_pixels(self):
        rgb = np.full((2, 2, 3), 25, dtype=np.uint8)
        reference = np.array([[0, 0], [1, 1]])
        prediction = np.array([[0, 1], [0, 1]])
        original = prediction.copy()
        overlay = error_overlay(rgb, reference, prediction, 1, ("FN",))
        expected = rgb.copy()
        expected[1, 0] = [41, 105, 255]
        np.testing.assert_array_equal(overlay, expected)
        np.testing.assert_array_equal(prediction, original)
