import unittest

import numpy as np

from src.evaluation import count_binary_pixels, pixel_metrics


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
