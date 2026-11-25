from dataclasses import dataclass

import numpy as np

from src.region_features import FEATURE_NAMES, _LEVELS, extract_region_features
from src.segmentation import segment_regions


@dataclass(frozen=True)
class RegionPrediction:
    """Resultado espacial e classes previstas, alinhadas pelos IDs das regiões."""

    mask: np.ndarray
    regions: np.ndarray
    region_ids: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True)
class RegionPredictionDetails(RegionPrediction):
    """Características e probabilidades de classe 1 alinhadas aos IDs previstos."""

    features: np.ndarray
    probabilities: np.ndarray


def reconstruct_binary_mask(
    regions: np.ndarray, region_ids: np.ndarray, predictions: np.ndarray
) -> np.ndarray:
    """Expande as previsões por ID e retorna uma máscara uint8 com valores 0 e 1."""
    if predictions.shape != region_ids.shape or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Cada ID de região deve ter exatamente uma previsão binária.")

    # Os índices compactos permitem recuperar cada pixel mesmo com IDs esparsos.
    spatial_ids, pixel_indices = np.unique(regions.ravel(), return_inverse=True)
    order = np.argsort(region_ids)
    if not np.array_equal(spatial_ids, region_ids[order]):
        raise ValueError("Os IDs das previsões não correspondem ao mapa de regiões.")
    pixel_classes = predictions[order][pixel_indices]
    return pixel_classes.reshape(regions.shape).astype(np.uint8, copy=False)


def predict_regions(
    artifact: dict, rgb: np.ndarray, *, include_details: bool = False,
) -> RegionPrediction:
    """Classifica todas as regiões e, opcionalmente, devolve os dados diagnósticos."""
    model = artifact["model"]
    config = artifact["pipeline_config"]
    if (tuple(artifact["feature_names"]) != FEATURE_NAMES
            or model.n_features_in_ != len(FEATURE_NAMES)):
        raise ValueError("As características do modelo são incompatíveis com o extrator.")
    feature_config = config["features"]
    if feature_config["quantization_levels"] != _LEVELS:
        raise ValueError("A quantização do modelo é incompatível com o extrator.")

    regions = segment_regions(rgb, **config["slic"])
    region_ids, features = extract_region_features(
        rgb, regions, neighborhood_radius=feature_config["neighborhood_radius"],
    )
    predictions = model.predict(features)
    mask = reconstruct_binary_mask(regions, region_ids, predictions)
    if include_details:
        positive_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
        if positive_columns.size != 1:
            raise ValueError("O modelo deve conter uma única coluna para a classe 1.")
        probabilities = model.predict_proba(features)[:, positive_columns[0]].copy()
        return RegionPredictionDetails(mask, regions, region_ids, predictions, features, probabilities)
    return RegionPrediction(mask, regions, region_ids, predictions)
