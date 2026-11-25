import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from experiments.evaluate_multiscale_context import train_variants
from src.feature_extractor import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS
from src.region_training import FOREST_CONFIG, train_region_forest


class RegionTrainingTests(unittest.TestCase):
    def setUp(self):
        workspace = self.enterContext(TemporaryDirectory())
        self.root = Path(workspace)
        self.rng = np.random.default_rng(123)
        self.labels = np.array([0, 1, -1, 1, 0, 1, -1, 0] * 3)

    def dataset(self, groups, image_id, *, legacy=False):
        names = [name for group in groups for name in FEATURE_GROUPS[group]]
        data = {
            "feature_names": np.array(names),
            "features": self.rng.random((self.labels.size, len(names))),
            "labels": self.labels,
            "image_ids": np.array([image_id]),
            "pipeline_config": np.array(json.dumps({"slic": {"n_segments": 800}})),
        }
        if not legacy:
            data["feature_groups"] = np.array(groups)
        return data

    def train(self, train, validation, model_name="model.joblib"):
        train_path = self.root / "train.npz"
        validation_path = self.root / "validation.npz"
        model_path = self.root / model_name
        np.savez(train_path, **train)
        np.savez(validation_path, **validation)
        report = train_region_forest(train_path, validation_path, model_path)
        return report, joblib.load(model_path)

    def test_training_configurations_and_model_metadata(self):
        configurations = (
            (DEFAULT_FEATURE_GROUPS, 15),
            ((*DEFAULT_FEATURE_GROUPS, "multiscale_lab"), 21),
            (tuple(FEATURE_GROUPS), 25),
            (("entropy", "multiscale_robust"), 5),
        )
        for groups, columns in configurations:
            with self.subTest(groups=groups):
                train = self.dataset(groups, "train", legacy=columns == 15)
                validation = self.dataset(groups, "validation")
                report, artifact = self.train(train, validation)
                self.assertEqual(artifact["model"].n_features_in_, columns)
                self.assertEqual(artifact["feature_groups"], list(groups))
                self.assertEqual(artifact["feature_names"], train["feature_names"].tolist())
                self.assertEqual(list(report["feature_importance"]), artifact["feature_names"])
                binary_count = int(np.count_nonzero(self.labels != -1))
                self.assertEqual(report["training_regions"], binary_count)
                self.assertEqual(report["validation_regions"], binary_count)
                self.assertEqual(artifact["model"].classes_.tolist(), [0, 1])

    def test_original_equivalence_and_reproducibility(self):
        train = self.dataset(DEFAULT_FEATURE_GROUPS, "train", legacy=True)
        validation = self.dataset(DEFAULT_FEATURE_GROUPS, "validation", legacy=True)
        binary = self.labels != -1
        original = RandomForestClassifier(**FOREST_CONFIG).fit(train["features"][binary], self.labels[binary])
        first_report, first = self.train(train, validation)
        second_report, second = self.train(train, validation, "second.joblib")
        samples = validation["features"][binary]
        for artifact in (first, second):
            model = artifact["model"]
            self.assertEqual(model.get_params(), original.get_params())
            np.testing.assert_array_equal(model.predict(samples), original.predict(samples))
            np.testing.assert_allclose(model.predict_proba(samples), original.predict_proba(samples))
        self.assertEqual(first_report["feature_importance"], second_report["feature_importance"])

    def test_rejects_incompatible_names_order_groups_and_pipeline(self):
        train = self.dataset(("entropy", "gradient"), "train")
        different = self.dataset(("entropy", "lab_mean"), "validation")
        reordered = self.dataset(("gradient", "entropy"), "validation")
        wrong_groups = self.dataset(("entropy", "gradient"), "validation")
        wrong_groups["feature_groups"] = np.array(["gradient", "entropy"])
        wrong_pipeline = self.dataset(("entropy", "gradient"), "validation")
        wrong_pipeline["pipeline_config"] = np.array("{}")
        for validation, message in (
            (different, "nomes ou a ordem"),
            (reordered, "nomes ou a ordem"),
            (wrong_groups, "Grupos incompatíveis"),
            (wrong_pipeline, "configurações"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.train(train, validation)
        self.assertFalse((self.root / "model.joblib").exists())

    def test_e01_uses_shared_columns_and_reuses_historical_model(self):
        data = self.dataset(tuple(FEATURE_GROUPS), "train", legacy=True)
        binary = self.labels != -1
        baseline = RandomForestClassifier(**FOREST_CONFIG).fit(
            data["features"][binary, :15], self.labels[binary],
        )
        artifact = {"model": baseline, "pipeline_config": json.loads(data["pipeline_config"].item())}
        variants = ("E01.1-A", "E01.1-B")
        models, _ = train_variants(data, artifact, self.root, variants)
        for name, columns, groups in (
            ("E01.1-A", 21, (*DEFAULT_FEATURE_GROUPS, "multiscale_lab")),
            ("E01.1-B", 25, tuple(FEATURE_GROUPS)),
        ):
            path = self.root / f"{name}.joblib"
            saved = joblib.load(path)
            self.assertEqual(saved["feature_names"], data["feature_names"][:columns].tolist())
            self.assertEqual(saved.pop("feature_groups"), list(groups))
            samples = data["features"][binary, :columns]
            original = RandomForestClassifier(**baseline.get_params()).fit(samples, self.labels[binary])
            np.testing.assert_array_equal(models[name].predict(samples), original.predict(samples))
            np.testing.assert_allclose(models[name].predict_proba(samples), original.predict_proba(samples))
            joblib.dump(saved, path)
        with patch("experiments.evaluate_multiscale_context.fit_region_forest") as fit:
            train_variants(data, artifact, self.root, variants, reuse=True)
        fit.assert_not_called()
        self.assertEqual(list(self.root.glob("*.npz")), [])
