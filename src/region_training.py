import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)

from src.region_features import FEATURE_NAMES


FOREST_CONFIG = {
    "n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2,
    "min_samples_split": 2, "max_features": "sqrt", "class_weight": "balanced",
    "random_state": 42, "n_jobs": -1,
}


def load_binary_samples(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Carrega somente as características e os rótulos binários usados pelo modelo."""
    with np.load(path, allow_pickle=False) as dataset:
        features, labels = dataset["features"], dataset["labels"]
        metadata = {
            "feature_names": dataset["feature_names"].tolist(),
            "pipeline_config": json.loads(dataset["pipeline_config"].item()),
            "image_ids": dataset["image_ids"].tolist(),
        }
    if (tuple(metadata["feature_names"]) != FEATURE_NAMES
            or features.shape != (labels.size, len(FEATURE_NAMES))
            or labels.ndim != 1 or not np.isfinite(features).all()
            or not np.isin(labels, (-1, 0, 1)).all()):
        raise ValueError(f"Características ou rótulos inválidos em {path}.")
    selected = labels != -1
    if np.unique(labels[selected]).size != 2:
        raise ValueError(f"São necessárias regiões positivas e negativas em {path}.")
    return features[selected], labels[selected], metadata


def train_region_forest(train_path: Path, validation_path: Path, model_path: Path) -> dict:
    """Ajusta o modelo no treino e avalia apenas regiões binárias da validação."""
    train_features, train_labels, train_metadata = load_binary_samples(train_path)
    validation_features, validation_labels, validation_metadata = load_binary_samples(validation_path)
    if set(train_metadata["image_ids"]) & set(validation_metadata["image_ids"]):
        raise ValueError("Há imagens em comum entre treino e validação.")
    if train_metadata["pipeline_config"] != validation_metadata["pipeline_config"]:
        raise ValueError("As configurações de treino e validação divergem.")

    model = RandomForestClassifier(**FOREST_CONFIG)
    started = perf_counter()
    model.fit(train_features, train_labels)
    training_seconds = perf_counter() - started
    predictions = model.predict(validation_features)
    joblib.dump({"model": model, "forest_config": FOREST_CONFIG, **train_metadata}, model_path,
                compress=3)

    return {
        "evaluation": "Regiões da validação, com exclusão das ambíguas",
        "training_regions": train_labels.size,
        "validation_regions": validation_labels.size,
        "confusion_matrix_rows_true_columns_predicted_0_1": (
            confusion_matrix(validation_labels, predictions, labels=[0, 1]).tolist()
        ),
        "accuracy": accuracy_score(validation_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(validation_labels, predictions),
        "precision": precision_score(validation_labels, predictions, zero_division=0),
        "recall": recall_score(validation_labels, predictions, zero_division=0),
        "f1": f1_score(validation_labels, predictions, zero_division=0),
        "feature_importance": dict(zip(FEATURE_NAMES, model.feature_importances_)),
        "training_seconds": training_seconds,
        "model_bytes": model_path.stat().st_size,
    }
