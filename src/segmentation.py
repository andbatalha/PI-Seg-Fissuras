import numpy as np
from skimage.segmentation import slic


def segment_regions(
    rgb: np.ndarray,
    *,
    n_segments: int = 800,
    compactness: float = 10,
    sigma: float = 1,
    start_label: int = 1,
) -> np.ndarray:
    """Segmenta uma imagem RGB (altura, largura, 3) em regiões SLIC."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("A imagem deve ter formato (altura, largura, 3).")

    return slic(
        rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=start_label,
        channel_axis=-1,
    )
