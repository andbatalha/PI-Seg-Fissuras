from dataclasses import dataclass

import numpy as np

from src.feature_extractor import FEATURE_GROUPS, canonical_feature_groups, extract_features
from src.region_features import _LEVELS
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


def model_feature_groups(artifact: dict) -> tuple[str, ...]:
    """Valida os grupos e a ordem das colunas a partir dos metadados do modelo."""
    try:
        names = tuple(artifact["feature_names"])
        if not names or any(not isinstance(name, str) for name in names):
            raise ValueError
        if "feature_groups" in artifact:
            stored_groups = artifact["feature_groups"]
            if isinstance(stored_groups, str):
                raise ValueError
            groups = tuple(stored_groups)
        else:
            # Modelos históricos registravam apenas os nomes em ordem.
            groups = tuple(group for group, columns in FEATURE_GROUPS.items() if columns[0] in names)
        if groups != canonical_feature_groups(groups):
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Não foi possível determinar os grupos de características do modelo.") from error
    if names != tuple(name for group in groups for name in FEATURE_GROUPS[group]):
        raise ValueError("Os nomes ou a ordem das características do modelo divergem do extrator.")
    if getattr(artifact["model"], "n_features_in_", None) != len(names):
        raise ValueError("A quantidade de características registrada diverge da esperada pelo estimador.")
    return groups


def _positive_column(model) -> int:
    """Localiza a coluna da classe 1 em um estimador binário."""
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or not np.isin((0, 1), classes).all():
        raise ValueError("O modelo deve fornecer as duas classes binárias 0 e 1.")
    return int(np.flatnonzero(classes == 1)[0])


def forest_decisions(model, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recupera a decisão do Random Forest e P(classe 1) em uma única passagem.

    O primeiro máximo na ordem de classes vence empates, como em predict().
    """
    positive_column = _positive_column(model)
    return probabilities.argmax(axis=1) == positive_column, probabilities[:, positive_column]


def predict_regions(
    artifact: dict, rgb: np.ndarray, *, include_details: bool = False, threshold: float | None = None,
) -> RegionPrediction:
    """Classifica regiões extraídas do RGB e reconstrói a máscara binária.

    Com threshold=None, usa predict() do estimador. Um limiar explícito
    classifica como positiva toda região com P(classe 1) >= threshold.
    """
    model = artifact["model"]
    groups = model_feature_groups(artifact)
    try:
        config = artifact["pipeline_config"]
        feature_config = config["features"]
        slic_config = config["slic"]
        radius = feature_config["neighborhood_radius"]
        levels = feature_config["quantization_levels"]
    except (KeyError, TypeError) as error:
        raise ValueError("Configuração SLIC ou de características incompleta no modelo.") from error
    if levels != _LEVELS:
        raise ValueError("A quantização do modelo é incompatível com o extrator.")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("O limiar de decisão deve estar entre 0 e 1.")

    regions = segment_regions(rgb, **slic_config)
    region_ids, features, names = extract_features(rgb, regions, groups, neighborhood_radius=radius)
    if names != tuple(artifact["feature_names"]):
        raise ValueError("Os nomes ou a ordem das características extraídas divergem do modelo.")
    if features.shape != (region_ids.size, model.n_features_in_):
        raise ValueError("A matriz extraída diverge da quantidade de regiões ou de colunas esperadas.")
    if threshold is None:
        predictions = model.predict(features)
    if include_details or threshold is not None:
        positive_column = _positive_column(model)
        probabilities = model.predict_proba(features)[:, positive_column].copy()
    if threshold is not None:
        predictions = (probabilities >= threshold).astype(np.uint8)
    mask = reconstruct_binary_mask(regions, region_ids, predictions)
    if include_details:
        return RegionPredictionDetails(mask, regions, region_ids, predictions, features, probabilities)
    return RegionPrediction(mask, regions, region_ids, predictions)
