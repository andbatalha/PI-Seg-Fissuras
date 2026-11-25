import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from src.feature_extractor import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS, extract_features
from src.region_dataset import PIPELINE_CONFIG, build_region_dataset
from src.region_features import FEATURE_NAMES, extract_region_features
from src.region_training import load_binary_samples


class RegionDatasetTests(unittest.TestCase):
    def test_configurations_preserve_metadata_alignment_and_ambiguous_regions(self):
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        regions = np.array([[90, 2, 9], [90, 2, 9]])
        annotation = np.array([[5, 5, 0], [5, 0, 0]], dtype=np.uint8)
        groups_a = (*DEFAULT_FEATURE_GROUPS, "multiscale_lab")
        groups_b = tuple(FEATURE_GROUPS)
        cases = (None, groups_a, groups_b, ("multiscale_robust", "entropy", "lab_mean"))
        with (
            TemporaryDirectory() as temporary,
            patch("src.region_dataset.load_image", return_value=rgb),
            patch("src.region_dataset.segment_regions", return_value=regions) as segment,
            patch("src.region_dataset.load_label", return_value=annotation),
        ):
            for index, groups in enumerate(cases):
                path = Path(temporary) / f"variant-{index}.npz"
                arguments = {"feature_groups": iter(groups)} if groups else {}
                result = build_region_dataset(Path(temporary), ["img_2", "img_1"], path, **arguments)
                ids, features, names = extract_features(rgb, regions, groups or DEFAULT_FEATURE_GROUPS)
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data["image_ids"].tolist(), ["img_2", "img_1"])
                    self.assertEqual(json.loads(data["pipeline_config"].item()), PIPELINE_CONFIG)
                    self.assertEqual(tuple(data["feature_names"]), names)
                    self.assertEqual(data["feature_groups"].tolist(),
                                     [group for group in FEATURE_GROUPS if group in (groups or DEFAULT_FEATURE_GROUPS)])
                    np.testing.assert_array_equal(data["features"], np.tile(features, (2, 1)))
                    np.testing.assert_array_equal(data["region_ids"], np.tile(ids, 2))
                    np.testing.assert_array_equal(data["pixel_counts"], [2] * 6)
                    np.testing.assert_array_equal(data["target_counts"], [1, 0, 2] * 2)
                    np.testing.assert_array_equal(data["labels"], [-1, 0, 1] * 2)
                    np.testing.assert_array_equal(data["target_fractions"], [0.5, 0, 1] * 2)
                    np.testing.assert_array_equal(data["image_indices"], [0, 0, 0, 1, 1, 1])
                    self.assertEqual(result["feature_shape"], (6, len(names)))
                    if groups is None:
                        baseline = extract_region_features(rgb, regions)[1]
                        np.testing.assert_array_equal(data["features"], np.tile(baseline, (2, 1)))
                        self.assertEqual(tuple(data["feature_names"]), FEATURE_NAMES)
                        previous = {key: data[key] for key in data.files if key != "feature_groups"}
                        legacy_path = Path(temporary) / "legacy.npz"
                        np.savez(legacy_path, **previous)
                        new, new_labels, metadata = load_binary_samples(path)
                        old, old_labels, old_metadata = load_binary_samples(legacy_path)
                        np.testing.assert_array_equal(new, old)
                        np.testing.assert_array_equal(new_labels, old_labels)
                        self.assertEqual(metadata, old_metadata)
            self.assertEqual(segment.call_args.kwargs, PIPELINE_CONFIG["slic"])

    def test_annotations_change_labels_without_changing_features(self):
        rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        regions = np.array([[20, 3], [20, 3]])
        annotation = np.array([[0, 5], [0, 5]], dtype=np.uint8)
        with (
            TemporaryDirectory() as temporary,
            patch("src.region_dataset.load_image", return_value=rgb),
            patch("src.region_dataset.segment_regions", return_value=regions),
            patch("src.region_dataset.load_label", side_effect=(annotation, 5 - annotation)),
        ):
            path = Path(temporary) / "contrasting_labels.npz"
            build_region_dataset(Path(temporary), ["before", "after"], path, feature_groups=["multiscale_lab"])
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["features"][:2], data["features"][2:])
                np.testing.assert_array_equal(data["labels"], [1, 0, 0, 1])
