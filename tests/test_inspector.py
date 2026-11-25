import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from src.evaluation import count_binary_pixels, pixel_metrics
from src.region_features import FEATURE_NAMES
from src.region_labeling import label_regions
from visualization import app


class InspectorTests(unittest.TestCase):
    def test_filters_navigation_and_visual_controls(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        prediction = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        positive = np.array([[0, 0], [0, 1]], dtype=np.uint8)
        negative = np.zeros((2, 2), dtype=np.uint8)
        regions = np.array([[10, 90], [10, 90]])
        click = None

        def display_image(*args, **kwargs):
            return click

        def image_data(model_path, signature, rgb_path, label_path, image_signatures):
            names = FEATURE_NAMES if model_path.parent == results else ("hist_entropy",)
            reference = negative if rgb_path.stem == "val_without_stain" else positive
            mask = prediction if model_path.parent == results else np.zeros_like(prediction)
            counts = count_binary_pixels(reference, mask)
            return {
                "rgb": rgb, "reference": reference, "prediction": mask,
                "boundaries": prediction.astype(bool), "region_count": 2,
                "counts": counts, "metrics": pixel_metrics(counts),
                "inference_seconds": 0.01,
                "regions": regions, "region_ids": np.array([10, 90]),
                "probabilities": np.array([0.5, 0.8]),
                "feature_names": names,
                "features": np.zeros((2, len(names))),
                "original_predictions": mask[0],
                "labeling": label_regions(regions, reference * 5),
                "labeling_config": {"negative_max": 0.1, "positive_min": 0.7},
            }

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            (results / "random_forest.joblib").touch()
            second = results / "manual" / "baseline" / "entropy" / "random_forest.joblib"
            second.parent.mkdir(parents=True)
            second.touch()
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
            def model_data(path, signature):
                names = ("hist_entropy",) if path == second else FEATURE_NAMES
                return {"pipeline_config": {"slic": {}}, "forest_config": {}, "feature_names": names}

            with (
                patch.multiple(app, ROOT=root, RESULTS=results),
                patch.object(app, "load_model", side_effect=model_data),
                patch.object(app, "inspect_image", side_effect=image_data) as inspect,
                patch("visualization.region_panels.streamlit_image_coordinates", side_effect=display_image),
            ):
                article = AppTest.from_string("from visualization.app import main\nmain()").run()

                def control(widgets, label):
                    return next(widget for widget in widgets if widget.label == label)

                self.assertFalse(article.exception or article.error)
                self.assertEqual(len(article.get("image")) or len(article.get("imgs")), 7)
                self.assertEqual(len(article.metric), 9)
                control(article.button, "Próxima").click().run()
                self.assertEqual(article.selectbox(key="image_id").value, "val_without_stain")
                control(article.selectbox, "Presença de Stain").set_value("Com Stain").run()
                self.assertEqual(article.selectbox(key="image_id").options, ["val_with_stain"])
                control(article.selectbox, "Presença de Stain").set_value("Todas").run()
                control(article.selectbox, "Ordenar por").set_value("IoU").run()
                self.assertEqual(article.selectbox(key="image_id").options, [
                    "val_with_stain", "val_without_stain",
                ])
                control(article.toggle, "Sobrepor máscaras ao RGB").set_value(True).run()
                control(article.toggle, "Mostrar fronteiras SLIC").set_value(True).run()
                control(article.slider, "Opacidade das sobreposições").set_value(40).run()
                click = {"x": 75, "y": 10, "width": 100, "height": 100}
                article.run()
                self.assertEqual(article.selectbox(key="superpixel_id").value, 90)
                click = None
                control(article.button, "Adicionar à comparação").click().run()
                self.assertEqual(article.multiselect(key="compared_regions").value, [90])
                article.multiselect(key="compared_regions").set_value([90, 10]).run()
                article.slider(key="threshold").set_value(1.0).run()
                self.assertEqual(article.multiselect(key="compared_regions").value, [90, 10])
                self.assertEqual(next(metric.value for metric in article.metric if metric.label == "TP"), "0")
                control(article.button, "Restaurar limiar padrão").click().run()
                self.assertEqual(article.slider(key="threshold").value, 0.5)
                control(article.button, "Limpar comparação").click().run()
                self.assertEqual(article.multiselect(key="compared_regions").value, [])
                article.multiselect(key="compared_regions").set_value([90]).run()
                control(article.radio, "Conjunto").set_value("Treinamento").run()
                self.assertEqual(article.selectbox(key="image_id").options, ["train_image"])
                self.assertEqual(article.multiselect(key="compared_regions").value, [])
                self.assertFalse(article.exception or article.error)
                control(article.radio, "Conjunto").set_value("Validação").run()
                article.multiselect(key="compared_regions").set_value([90]).run()
                control(article.selectbox, "Modelo ativo").set_value(second).run()
                self.assertEqual(inspect.call_args.args[0], second)
                self.assertEqual(control(article.selectbox, "Ordenar por").options, ["ID"])
                self.assertEqual(article.multiselect(key="compared_regions").value, [])
                self.assertEqual(next(metric.value for metric in article.metric if metric.label == "TP"), "0")
                self.assertTrue(any(list(frame.value["Característica"]) == ["hist_entropy"]
                                    for frame in article.dataframe if "Característica" in frame.value))
                self.assertFalse(article.exception or article.error)
