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

from src.feature_extractor import FEATURE_GROUPS


FOREST_CONFIG = {
    "n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2,
    "min_samples_split": 2, "max_features": "sqrt", "class_weight": "balanced",
    "random_state": 42, "n_jobs": -1,
}


def load_binary_samples(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Carrega as amostras binárias e os metadados de um dataset NPZ."""
    with np.load(path, allow_pickle=False) as dataset:
        return binary_samples(dataset, source=path)


def binary_samples(dataset, *, source: Path | str = "dataset") -> tuple[np.ndarray, np.ndarray, dict]:
    """Valida as colunas e remove regiões ambíguas de um NPZ ou dicionário."""
    features, labels = dataset["features"], dataset["labels"]
    names = dataset["feature_names"].tolist()
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"Nomes de características inválidos em {source}.")
    if (features.shape != (labels.size, len(names))
            or labels.ndim != 1 or not np.isfinite(features).all()
            or not np.isin(labels, (-1, 0, 1)).all()):
        raise ValueError(f"Características ou rótulos inválidos em {source}.")
    if "feature_groups" in dataset:
        groups = dataset["feature_groups"].tolist()
        if (not isinstance(groups, list) or not groups
                or any(not isinstance(group, str) or group not in FEATURE_GROUPS for group in groups)
                or len(set(groups)) != len(groups)
                or [name for group in groups for name in FEATURE_GROUPS[group]] != names):
            raise ValueError(f"Grupos incompatíveis com os nomes ou a ordem das características em {source}.")
    else:
        # Um NPZ antigo só identifica grupos quando contém blocos completos na ordem gravada.
        groups = []
        position = 0
        while position < len(names):
            for group, columns in FEATURE_GROUPS.items():
                if tuple(names[position:position + len(columns)]) == columns:
                    groups.append(group)
                    position += len(columns)
                    break
            else:
                groups = []
                break
    metadata = {
        "feature_names": names,
        "feature_groups": groups,
        "pipeline_config": json.loads(dataset["pipeline_config"].item()),
        "image_ids": dataset["image_ids"].tolist(),
    }
    selected = labels != -1
    if np.unique(labels[selected]).size != 2:
        raise ValueError(f"São necessárias regiões positivas e negativas em {source}.")
    return features[selected], labels[selected], metadata


def fit_region_forest(features: np.ndarray, labels: np.ndarray, *,
                      forest_config: dict = FOREST_CONFIG) -> tuple[RandomForestClassifier, float]:
    """Ajusta o Random Forest em amostras binárias já validadas e mede o tempo do ajuste."""
    model = RandomForestClassifier(**forest_config)
    started = perf_counter()
    model.fit(features, labels)
    return model, perf_counter() - started


def train_region_forest(train_path: Path, validation_path: Path, model_path: Path) -> dict:
    """Ajusta o modelo no treino e avalia apenas regiões binárias da validação."""
    train_features, train_labels, train_metadata = load_binary_samples(train_path)
    validation_features, validation_labels, validation_metadata = load_binary_samples(validation_path)
    if train_metadata["feature_names"] != validation_metadata["feature_names"]:
        raise ValueError("Os nomes ou a ordem das características de treino e validação divergem.")
    if train_metadata["feature_groups"] != validation_metadata["feature_groups"]:
        raise ValueError("Os grupos de características de treino e validação divergem.")
    if set(train_metadata["image_ids"]) & set(validation_metadata["image_ids"]):
        raise ValueError("Há imagens em comum entre treino e validação.")
    if train_metadata["pipeline_config"] != validation_metadata["pipeline_config"]:
        raise ValueError("As configurações de treino e validação divergem.")

    model, training_seconds = fit_region_forest(train_features, train_labels)
    predictions = model.predict(validation_features)
    joblib.dump({"model": model, "forest_config": FOREST_CONFIG, **train_metadata}, model_path,
                compress=3)

    return {
        "evaluation": "Regiões da validação, com exclusão das ambíguas",
        "configuration": {"feature_groups": train_metadata["feature_groups"],
                          "feature_count": len(train_metadata["feature_names"])},
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
        "feature_importance": dict(zip(train_metadata["feature_names"], model.feature_importances_)),
        "training_seconds": training_seconds,
        "model_bytes": model_path.stat().st_size,
    }
