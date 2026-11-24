import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from src.evaluation import count_binary_pixels, pixel_metrics
from visualization import app


class InspectorTests(unittest.TestCase):
    def test_filters_navigation_and_visual_controls(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        prediction = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        positive = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        negative = np.zeros((2, 2), dtype=np.uint8)

        def image_data(model_path, signature, rgb_path, label_path, image_signatures):
            reference = negative if rgb_path.stem == "val_without_stain" else positive
            counts = count_binary_pixels(reference, prediction)
            return {
                "rgb": rgb, "reference": reference, "prediction": prediction,
                "boundaries": prediction.astype(bool), "region_count": 2,
                "counts": counts, "metrics": pixel_metrics(counts),
                "inference_seconds": 0.01,
            }

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            (results / "random_forest.joblib").touch()
            (results / "split.json").write_text(json.dumps({
                "train": ["train_image"],
                "validation": ["val_without_stain", "val_with_stain"],
                "test_reserved": ["reserved"],
            }), encoding="utf-8")
            report = []
            for name, reference in (("val_without_stain", negative), ("val_with_stain", positive)):
                counts = count_binary_pixels(reference, prediction)
                report.append({"image_id": name, "has_stain": bool(reference.any()),
                               "counts": counts, "metrics": pixel_metrics(counts)})
            (results / "pixel_validation_per_image.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            model = {"pipeline_config": {"slic": {}}, "forest_config": {}, "feature_names": []}
            with (
                patch.multiple(app, ROOT=root, RESULTS=results),
                patch.object(app, "load_model", return_value=model),
                patch.object(app, "inspect_image", side_effect=image_data),
            ):
                article = AppTest.from_string("from visualization.app import main\nmain()").run()
                self.assertFalse(article.exception or article.error)
                self.assertEqual(len(article.get("image")) or len(article.get("imgs")), 4)
                self.assertEqual(len(article.metric), 9)
                article.button[1].click().run()
                self.assertEqual(article.selectbox(key="image_id").value, "val_without_stain")
                article.selectbox[0].set_value("Com Stain").run()
                self.assertEqual(article.selectbox(key="image_id").options, ["val_with_stain"])
                article.selectbox[0].set_value("Todas").run()
                article.selectbox[1].set_value("IoU").run()
                self.assertEqual(article.selectbox(key="image_id").options, [
                    "val_with_stain", "val_without_stain",
                ])
                article.toggle[0].set_value(True).run()
                article.toggle[1].set_value(True).run()
                article.slider[0].set_value(40).run()
                article.radio[0].set_value("Treinamento").run()
                self.assertEqual(article.selectbox(key="image_id").options, ["train_image"])
                self.assertFalse(article.exception or article.error)
