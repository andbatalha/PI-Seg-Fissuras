from dataclasses import dataclass

import numpy as np

from src.dataset.classes import DamageClass
from src.dataset.masks import create_binary_mask


@dataclass(frozen=True)
class RegionLabeling:
    """Vetores alinhados por ID; pixel_indices mapeia cada pixel à sua linha."""

    region_ids: np.ndarray
    pixel_counts: np.ndarray
    target_counts: np.ndarray
    target_fractions: np.ndarray
    labels: np.ndarray
    pixel_indices: np.ndarray


def label_regions(
    regions: np.ndarray,
    label: np.ndarray,
    *,
    class_id: int = DamageClass.STAIN,
    negative_max: float = 0.10,
    positive_min: float = 0.70,
) -> RegionLabeling:
    """Rotula regiões como negativas (0), positivas (1) ou ambíguas (-1)."""
    if regions.ndim != 2 or label.ndim != 2 or regions.shape != label.shape:
        raise ValueError("A segmentação e a label devem ser matrizes 2D de mesmo formato.")
    if regions.size == 0 or not np.issubdtype(regions.dtype, np.integer):
        raise ValueError("A segmentação deve conter identificadores inteiros de região.")
    if not 0 <= negative_max < positive_min <= 1:
        raise ValueError("Os limiares devem satisfazer 0 <= negative_max < positive_min <= 1.")

    target_mask = create_binary_mask(label, DamageClass(class_id)).ravel().astype(bool)
    region_ids, pixel_indices, pixel_counts = np.unique(
        regions.ravel(), return_inverse=True, return_counts=True
    )
    target_counts = np.bincount(pixel_indices[target_mask], minlength=region_ids.size)
    target_fractions = target_counts / pixel_counts

    labels = np.full(region_ids.size, -1, dtype=np.int8)
    labels[target_fractions <= negative_max] = 0
    labels[target_fractions >= positive_min] = 1

    return RegionLabeling(
        region_ids, pixel_counts, target_counts, target_fractions, labels,
        pixel_indices.reshape(regions.shape),
    )
