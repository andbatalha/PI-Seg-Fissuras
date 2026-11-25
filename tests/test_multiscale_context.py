import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from skimage.color import rgb2lab

from experiments.evaluate_multiscale_context import check_alignment, prepare_split, validate_prepared
from src.feature_extractor import FEATURE_GROUPS, extract_features
from src.multiscale_context import MULTISCALE_NAMES, extract_multiscale_context, region_adjacency, region_neighborhoods
from src.region_features import FEATURE_NAMES, extract_region_features
from src.region_labeling import label_regions


class MultiscaleContextTests(unittest.TestCase):
    def test_graph_preserves_ids_and_ignores_diagonal_contacts(self):
        graph = region_adjacency(np.array([[2, 11], [23, 29]]))
        self.assertEqual(graph, {2: {11, 23}, 11: {2, 29}, 23: {2, 29}, 29: {11, 23}})
        self.assertEqual(region_neighborhoods(graph, 2), ({11, 23}, {11, 23, 29}, {11, 23, 29}))
        chain = region_adjacency(np.array([[7, 9, 30, 31, 92, 200]]))
        self.assertEqual(region_neighborhoods(chain, 7), ({9}, {9, 30}, {9, 30, 31, 92}))

    def test_weighted_lab_means_and_exact_pixel_percentiles(self):
        pixel_colors = [[60, 80, 90], [35, 50, 20], [200, 90, 25], [220, 40, 25],
                        [120, 130, 20], [100, 70, 35], [250, 30, 40]]
        rgb = np.array([pixel_colors], dtype=np.uint8)
        regions = np.array([[10, 20, 30, 30, 70, 90, 200]])
        ids, extras = extract_multiscale_context(rgb, regions)
        np.testing.assert_array_equal(ids, [10, 20, 30, 70, 90, 200])
        lab = rgb2lab(rgb)[0]
        for scale, stop in enumerate((4, 6)):
            neighborhood = lab[1:stop]
            np.testing.assert_allclose(extras[0, 3*scale:3*scale+3], lab[0] - neighborhood.mean(axis=0))
            lower, upper = np.percentile(neighborhood[:, 0], [25, 75])
            self.assertAlmostEqual(extras[0, 6 + scale], lab[0, 0] - upper)
            self.assertAlmostEqual(extras[0, 8 + scale], upper - lower)
        self.assertTrue(np.isfinite(extras).all())

    def test_empty_neighborhood_and_original_contract(self):
        ids, features = extract_multiscale_context(np.zeros((2, 2, 3), dtype=np.uint8), np.full((2, 2), 77))
        np.testing.assert_array_equal(ids, [77])
        np.testing.assert_array_equal(features, np.zeros((1, 10)))
        self.assertEqual(len(FEATURE_NAMES), 15)
        self.assertFalse(set(FEATURE_NAMES) & set(MULTISCALE_NAMES))

    def test_dataset_preserves_original_columns_and_extracts_before_annotation(self):
        regions = np.array([[2, 93]])
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        annotation = np.array([[0, 5]], dtype=np.uint8)
        labeled = label_regions(regions, annotation)
        config = {"slic": {}, "labeling": {}, "features": {"neighborhood_radius": 5}}
        original = {"image_ids": np.array(["example"]), "image_indices": np.array([0, 0]),
                    "region_ids": labeled.region_ids, "pixel_counts": labeled.pixel_counts,
                    "target_counts": labeled.target_counts, "target_fractions": labeled.target_fractions,
                    "labels": labeled.labels, "features": extract_region_features(rgb, regions)[1],
                    "feature_names": np.array(FEATURE_NAMES),
                    "pipeline_config": np.array(json.dumps(config))}
        check_alignment(original, np.arange(2), original)
        with self.assertRaises(ValueError):
            check_alignment(original, np.arange(2), {**original, "region_ids": labeled.region_ids[::-1]})
        ordered = Mock()
        ordered.extract.side_effect = extract_features
        ordered.label.return_value = annotation
        prefix = "src.region_dataset"
        with TemporaryDirectory() as directory:
            source, output = (Path(directory) / name for name in ("base.npz", "new.npz"))
            np.savez(source, **original)
            with (patch(f"{prefix}.load_image", return_value=rgb),
                  patch(f"{prefix}.segment_regions", return_value=regions),
                  patch(f"{prefix}.extract_features", side_effect=ordered.extract),
                  patch(f"{prefix}.load_label", side_effect=ordered.label)):
                prepare_split(Path(directory), source, output, ["example"], config, validation=False)
                self.assertEqual(ordered.extract.call_args.args[2], ("multiscale_lab", "multiscale_robust"))
                prepare_split(Path(directory), source, output, ["example"], config, validation=True)
                self.assertEqual(ordered.extract.call_args.args[2], tuple(FEATURE_GROUPS))
            with np.load(output) as result:
                validate_prepared(dict(result.items()), original)
                self.assertEqual(result["features"].shape, (2, 25))
                np.testing.assert_array_equal(result["class_pixel_counts"], np.eye(6, dtype=int)[[0, 5]])
                self.assertEqual(result["feature_groups"].tolist(), list(FEATURE_GROUPS))
        self.assertEqual([call[0] for call in ordered.mock_calls], ["extract", "label", "extract", "label"])
