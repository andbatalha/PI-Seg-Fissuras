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


def predict_regions(artifact: dict, rgb: np.ndarray) -> RegionPrediction:
    """Segmenta o RGB e classifica todas as regiões com a configuração salva."""
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
    return RegionPrediction(mask, regions, region_ids, predictions)
