from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.feature import graycoprops
from skimage.morphology import disk


FEATURE_GROUPS = {
    "A": ("mean_l", "mean_a", "mean_b", "std_l", "std_a", "std_b"),
    "B": ("delta_l", "delta_a", "delta_b"),
    "C": ("hist_entropy",),
    "D": ("glcm_contrast", "glcm_homogeneity", "glcm_asm", "glcm_correlation"),
    "E": ("mean_gradient",),
}
FEATURE_NAMES = tuple(name for group in FEATURE_GROUPS.values() for name in group)
_LEVELS = 32
_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))


def _masked_glcm_features(levels, mask):
    """Sem pares, retorna quatro zeros. Correlação de variância nula vale 1."""
    height, width = mask.shape
    matrices = []
    for dy, dx in _OFFSETS:
        if height <= dy or width <= abs(dx):
            continue
        source = np.s_[:height - dy, max(0, -dx):width - max(0, dx)]
        destination = np.s_[dy:, max(0, dx):width - max(0, -dx)]
        valid = mask[source] & mask[destination]
        if not valid.any():
            continue
        pairs = levels[source][valid] * _LEVELS + levels[destination][valid]
        counts = np.bincount(pairs, minlength=_LEVELS**2).reshape(_LEVELS, _LEVELS)
        symmetric = counts + counts.T
        matrices.append(symmetric / symmetric.sum())

    if not matrices:
        return np.zeros(4)
    glcm = np.stack(matrices, axis=-1)[:, :, None, :]
    return np.array([
        graycoprops(glcm, prop).mean()
        for prop in ("contrast", "homogeneity", "ASM", "correlation")
    ])


def extract_region_features(
    rgb: np.ndarray,
    regions: np.ndarray,
    *,
    neighborhood_radius: int = 5,
    timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna IDs crescentes e uma matriz com colunas na ordem de FEATURE_NAMES.

    RGB segue a convenção do scikit-image: inteiros na escala do tipo ou floats
    em [0, 1]. Se fornecido, timings recebe tempos em segundos por grupo, além
    da preparação compartilhada e do total.
    """
    started = perf_counter()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != regions.shape:
        raise ValueError("A imagem RGB e a matriz de regiões devem ter as mesmas dimensões espaciais.")
    if regions.size == 0 or not np.issubdtype(regions.dtype, np.integer):
        raise ValueError("A segmentação deve conter identificadores inteiros de região.")
    if not isinstance(neighborhood_radius, (int, np.integer)) or neighborhood_radius < 0:
        raise ValueError("O raio da vizinhança deve ser um inteiro não negativo.")
    if np.issubdtype(rgb.dtype, np.floating):
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Imagens RGB de ponto flutuante devem estar em [0, 1].")
    elif not np.issubdtype(rgb.dtype, np.integer) or rgb.min() < 0:
        raise ValueError("A imagem RGB deve conter intensidades numéricas não negativas.")

    lab = rgb2lab(rgb.astype(np.float64) if rgb.dtype == np.float16 else rgb)
    lightness = np.clip(lab[..., 0], 0, 100)
    quantized = np.minimum((lightness * (_LEVELS / 100)).astype(np.intp), _LEVELS - 1)
    ids, inverse, counts = np.unique(regions.ravel(), return_inverse=True, return_counts=True)
    index_map = inverse.reshape(regions.shape)
    boxes = ndi.find_objects(index_map + 1)
    features = np.zeros((ids.size, len(FEATURE_NAMES)))
    group_started = perf_counter()
    elapsed = {"preparation": group_started - started}

    for channel in range(3):
        pixels = lab[..., channel].ravel()
        mean = np.bincount(inverse, weights=pixels) / counts
        residual = pixels - mean[inverse]
        features[:, channel] = mean
        features[:, channel + 3] = np.sqrt(np.bincount(inverse, weights=residual**2) / counts)
    elapsed["A"] = perf_counter() - group_started

    group_started = perf_counter()
    footprint = disk(neighborhood_radius)
    height, width = regions.shape
    for row, (ys, xs) in enumerate(boxes):
        crop = np.s_[max(0, ys.start - neighborhood_radius):min(height, ys.stop + neighborhood_radius),
                     max(0, xs.start - neighborhood_radius):min(width, xs.stop + neighborhood_radius)]
        mask = index_map[crop] == row
        outside = ndi.binary_dilation(mask, structure=footprint) & ~mask
        if outside.any():
            features[row, 6:9] = features[row, :3] - lab[crop][outside].mean(axis=0)
    elapsed["B"] = perf_counter() - group_started

    group_started = perf_counter()
    histogram = np.bincount(inverse * _LEVELS + quantized.ravel(), minlength=ids.size * _LEVELS)
    probabilities = histogram.reshape(-1, _LEVELS) / counts[:, None]
    log_probabilities = np.zeros_like(probabilities)
    np.log2(probabilities, out=log_probabilities, where=probabilities > 0)
    features[:, 9] = -(probabilities * log_probabilities).sum(axis=1)
    elapsed["C"] = perf_counter() - group_started

    group_started = perf_counter()
    for row, box in enumerate(boxes):
        features[row, 10:14] = _masked_glcm_features(quantized[box], index_map[box] == row)
    elapsed["D"] = perf_counter() - group_started

    group_started = perf_counter()
    normalized = lightness / 100
    gradient = np.hypot(ndi.sobel(normalized, axis=0), ndi.sobel(normalized, axis=1))
    features[:, 14] = np.bincount(inverse, weights=gradient.ravel()) / counts
    elapsed["E"] = perf_counter() - group_started
    if timings is not None:
        timings.update(elapsed, total=perf_counter() - started)
    return ids, features
