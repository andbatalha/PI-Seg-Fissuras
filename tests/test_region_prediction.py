import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.feature_extractor import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS
from src.multiscale_context import MULTISCALE_NAMES, extract_multiscale_context
from src.region_features import FEATURE_NAMES, extract_region_features
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
            with patch("src.region_prediction.extract_features", return_value=(ids, features, FEATURE_NAMES)) as extract:
                result = predict_regions(artifact, rgb)
        segment.assert_called_once_with(rgb, **slic)
        extract.assert_called_once_with(rgb, regions, DEFAULT_FEATURE_GROUPS, neighborhood_radius=3)
        model.predict.assert_called_once_with(features)
        np.testing.assert_array_equal(result.region_ids, ids)
        np.testing.assert_array_equal(result.mask, [[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            predict_regions({**artifact, "feature_names": FEATURE_NAMES[::-1]}, rgb)

    def test_historical_and_configurable_features_match_original_extractors(self):
        rgb = np.random.default_rng(42).integers(0, 256, size=(16, 18, 3), dtype=np.uint8)
        regions = np.repeat(np.repeat([[2, 90, 500], [701, 921, 4000]], 8, axis=0), 6, axis=1)
        ids, baseline = extract_region_features(rgb, regions)
        context_ids, context = extract_multiscale_context(rgb, regions)
        np.testing.assert_array_equal(ids, context_ids)
        all_names = (*FEATURE_NAMES, *MULTISCALE_NAMES)
        all_features = np.column_stack((baseline, context))
        arbitrary_groups = ("entropy", "multiscale_robust")
        arbitrary_names = tuple(name for group in arbitrary_groups for name in FEATURE_GROUPS[group])
        for names, groups in (
            (FEATURE_NAMES, None),
            (all_names[:21], None),
            (all_names, None),
            (arbitrary_names, arbitrary_groups),
        ):
            with self.subTest(columns=len(names)):
                model = Mock(n_features_in_=len(names), classes_=np.array([0, 1]))
                model.predict.return_value = np.array([0, 1, 0, 1, 0, 1])
                model.predict_proba.return_value = np.tile([0.4, 0.6], (len(ids), 1))
                artifact = self.artifact(model, names)
                if groups:
                    artifact["feature_groups"] = groups
                with patch("src.region_prediction.segment_regions", return_value=regions):
                    result = predict_regions(artifact, rgb, include_details=True)
                expected = all_features[:, [all_names.index(name) for name in names]]
                np.testing.assert_array_equal(result.features, expected)
                np.testing.assert_array_equal(result.region_ids, ids)
                np.testing.assert_array_equal(result.mask, np.isin(regions, ids[1::2]))
                np.testing.assert_array_equal(result.probabilities, np.full(len(ids), 0.6))

    def test_default_tie_and_explicit_threshold(self):
        rgb = np.zeros((1, 3, 3))
        regions = np.array([[20, 99, 700]])
        model = Mock(n_features_in_=1, classes_=np.array([0, 1]))
        model.predict.return_value = np.array([0, 0, 1])
        model.predict_proba.return_value = np.array([[1, 0], [0.5, 0.5], [0, 1]])
        artifact = self.artifact(model, ("hist_entropy",))
        with (
            patch("src.region_prediction.segment_regions", return_value=regions),
            patch("src.region_prediction.extract_features",
                  return_value=(regions[0], np.zeros((3, 1)), ("hist_entropy",))),
        ):
            default = predict_regions(artifact, rgb)
            model.predict_proba.assert_not_called()
            np.testing.assert_array_equal(default.mask, [[0, 0, 1]])
            for threshold, expected in ((0, [1, 1, 1]), (0.5, [0, 1, 1]), (1, [0, 0, 1])):
                adjusted = predict_regions(artifact, rgb, threshold=threshold)
                np.testing.assert_array_equal(adjusted.mask, [expected])
            model.predict.assert_called_once()
            with self.assertRaisesRegex(ValueError, "limiar"):
                predict_regions(artifact, rgb, threshold=np.nan)

    def test_metadata_and_extracted_columns_must_match(self):
        rgb = np.zeros((1, 1, 3))
        model = Mock(n_features_in_=1)
        artifact = self.artifact(model, ("hist_entropy",))
        malformed = (
            {**artifact, "feature_groups": ["gradient"]},
            {**artifact, "feature_groups": []},
            {**artifact, "feature_names": ["unknown"]},
            {**artifact, "model": Mock(n_features_in_=2)},
        )
        with patch("src.region_prediction.segment_regions", return_value=np.array([[9]])) as segment:
            for invalid in malformed:
                with self.assertRaises(ValueError):
                    predict_regions(invalid, rgb)
            segment.assert_not_called()
            for matrix, names in ((np.zeros((1, 1)), ("wrong",)), (np.zeros((1, 2)), ("hist_entropy",))):
                with patch("src.region_prediction.extract_features",
                           return_value=(np.array([9]), matrix, names)):
                    with self.assertRaises(ValueError):
                        predict_regions(artifact, rgb)
        model.predict.assert_not_called()

    @staticmethod
    def artifact(model, names):
        return {"model": model, "feature_names": names, "pipeline_config": {
            "slic": {}, "features": {"quantization_levels": 32, "neighborhood_radius": 5},
        }}
