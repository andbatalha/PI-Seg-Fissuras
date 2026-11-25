import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from experiments.evaluate_multiscale_context import score_model
from experiments.evaluate_regions import evaluate_image, evaluate_validation
from experiments.reporting import render_summary
from src.evaluation import count_binary_pixels, count_region_pixels, pixel_metrics
from src.region_prediction import reconstruct_binary_mask


class EvaluationTests(unittest.TestCase):
    def test_counts_and_metrics(self):
        reference = np.array([[0, 0, 0], [1, 1, 1]])
        prediction = np.array([[0, 0, 1], [0, 1, 1]])
        counts = count_binary_pixels(reference, prediction)
        self.assertEqual(counts, {"TP": 2, "FP": 1, "FN": 1, "TN": 2})
        self.assertEqual(pixel_metrics(counts), {
            "Accuracy": 2 / 3, "Precision": 2 / 3, "Recall": 2 / 3,
            "IoU": 1 / 2, "F1": 2 / 3,
        })

    def test_images_without_positive_reference(self):
        reference = np.zeros((2, 2), dtype=np.uint8)
        for prediction, expected_accuracy in ((reference, 1.0), (np.eye(2), 0.5)):
            with self.subTest(accuracy=expected_accuracy):
                self.assertEqual(pixel_metrics(count_binary_pixels(reference, prediction)), {
                    "Accuracy": expected_accuracy, "Precision": 0.0, "Recall": 0.0,
                    "IoU": 0.0, "F1": 0.0,
                })

    def test_rejects_mask_shape_mismatch(self):
        with self.assertRaises(ValueError):
            count_binary_pixels(np.zeros((2, 2)), np.zeros((1, 2)))

    def test_regional_counts_match_mask_counts(self):
        regions = np.array([[3, 3, 50], [700, 700, 50]])
        ids = np.array([700, 50, 3])
        reference = np.array([[0, 1, 1], [0, 0, 1]])
        area = np.array([np.count_nonzero(regions == identifier) for identifier in ids])
        stain = np.array([np.count_nonzero(reference[regions == identifier]) for identifier in ids])
        for reference_mask, positive in ((reference, stain), (np.zeros_like(reference), stain * 0)):
            for decisions in (np.array([0, 1, 1]), np.array([True, False, False]), np.zeros(3)):
                prediction = reconstruct_binary_mask(regions, ids, decisions)
                direct_counts = count_binary_pixels(reference_mask, prediction)
                regional_counts = count_region_pixels(positive, area - positive, decisions)
                self.assertEqual(regional_counts, direct_counts)
                self.assertEqual(pixel_metrics(regional_counts), pixel_metrics(direct_counts))
        empty = np.array([], dtype=int)
        self.assertEqual(pixel_metrics(count_region_pixels(empty, empty, empty))["Accuracy"], 0.0)

    def test_evaluator_reads_annotation_only_after_prediction(self):
        rgb = np.zeros((1, 2, 3))
        mask = np.array([[0, 1]])
        artifact = {}
        calls = Mock()
        calls.predict.return_value.mask = mask
        calls.label.return_value = np.array([[0, 5]], dtype=np.uint8)
        with (
            patch("experiments.evaluate_regions.load_image", return_value=rgb),
            patch("experiments.evaluate_regions.predict_regions", calls.predict),
            patch("experiments.evaluate_regions.load_label", calls.label),
        ):
            record, _ = evaluate_image(artifact, Path("BFDD"), "synthetic", threshold=0.8)
        self.assertEqual([call[0] for call in calls.mock_calls], ["predict", "label"])
        calls.predict.assert_called_once_with(artifact, rgb, threshold=0.8)
        self.assertEqual(record["counts"], {"TP": 1, "TN": 1, "FP": 0, "FN": 0})

    def test_cached_evaluation_scores_prepared_rows_and_preserves_ties(self):
        features = np.arange(12).reshape(3, 4)
        model = Mock(n_features_in_=2, classes_=np.array([0, 1]))
        model.predict_proba.side_effect = [np.array([[0.5, 0.5], [0.1, 0.9]]), np.array([[0.2, 0.8]])]
        locations = [np.array([0, 2]), np.array([1])]
        predictions, probabilities, times = score_model(model, {"features": features}, 2, locations)
        np.testing.assert_array_equal(predictions, [False, True, True])
        np.testing.assert_array_equal(probabilities, [0.5, 0.8, 0.9])
        self.assertEqual(times.shape, (2,))
        self.assertEqual(model.predict_proba.call_count, 2)
        for call, rows in zip(model.predict_proba.call_args_list, locations):
            np.testing.assert_array_equal(call.args[0], features[rows, :2])
        with self.assertRaisesRegex(ValueError, "colunas"):
            score_model(model, {"features": features}, 3, locations)

    def test_optimal_validation_uses_one_global_cut_without_repeating_inference(self):
        regions = np.array([[7, 30]])
        rgb = np.zeros((1, 2, 3))
        results = [Mock(regions=regions, region_ids=regions[0], probabilities=np.array(probabilities))
                   for probabilities in ([0.8, 0.3], [0.7, 0.2])]
        images = [(rgb, np.array([label]), result, 0.05)
                  for label, result in zip(([5, 0], [0, 0]), results)]
        artifact = {"model": Mock(n_features_in_=1), "feature_names": ["hist_entropy"],
                    "pipeline_config": {"labeling": {"class_id": 5}}}
        with (
            TemporaryDirectory() as directory,
            patch("experiments.evaluate_regions.infer_image", side_effect=images) as infer,
            patch("experiments.evaluate_regions.visualize_prediction") as draw,
        ):
            output = Path(directory)
            evaluate_validation(artifact, output, ["stain", "clean"], output,
                                optimize=True, model_path=output / "random_forest.joblib")
            report = json.loads((output / "pixel_validation_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["configuration"]["threshold"], 0.8)
            self.assertEqual(report["counts"], {"TP": 1, "TN": 3, "FP": 0, "FN": 0})
            self.assertEqual(report["mean_iou_with_stain"], 1.0)
            self.assertEqual(report["median_iou_with_stain"], 1.0)
            self.assertTrue((output / "summary.md").exists())
            summary = render_summary(output)
            self.assertIn("Limiar que maximiza a IoU neste conjunto de validação", summary)
            self.assertIn("Características (1) · entropy", summary)
            self.assertEqual(infer.call_count, 2)
            np.testing.assert_array_equal(draw.call_args_list[0].args[2], [[1, 0]])
