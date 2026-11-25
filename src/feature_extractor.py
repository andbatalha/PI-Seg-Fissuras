"""Seleciona grupos dos descritores regionais já implementados no projeto."""

from collections.abc import Iterable
from time import perf_counter

import numpy as np
from skimage.color import rgb2lab

from src.multiscale_context import MULTISCALE_NAMES, extract_multiscale_context
from src.region_features import (
    BASE_FEATURE_COLUMNS,
    FEATURE_NAMES,
    _validate_rgb_regions,
    extract_region_features,
)


FEATURE_GROUPS = {
    **{group: FEATURE_NAMES[columns] for group, columns in BASE_FEATURE_COLUMNS.items()},
    "multiscale_lab": MULTISCALE_NAMES[:6],
    "multiscale_robust": MULTISCALE_NAMES[6:],
}
DEFAULT_FEATURE_GROUPS = tuple(BASE_FEATURE_COLUMNS)


def canonical_feature_groups(groups: Iterable[str]) -> tuple[str, ...]:
    """Valida a seleção e devolve os grupos na ordem das colunas."""
    selected = {groups} if isinstance(groups, str) else set(groups)
    if not selected:
        raise ValueError("Selecione pelo menos um grupo de características.")
    unknown = selected.difference(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Grupos de características inválidos: {', '.join(sorted(map(str, unknown)))}.")
    return tuple(group for group in FEATURE_GROUPS if group in selected)


def extract_features(
    rgb: np.ndarray,
    regions: np.ndarray,
    groups: Iterable[str],
    *,
    neighborhood_radius: int = 5,
    timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Retorna IDs crescentes, matriz e nomes na ordem canônica dos grupos.

    A ordem de entrada e grupos repetidos não afetam a ordem ou o número de
    colunas. A imagem e os IDs seguem as convenções do extrator original.
    """
    selected = canonical_feature_groups(groups)
    base_groups = set(selected).intersection(BASE_FEATURE_COLUMNS)
    include_lab = "multiscale_lab" in selected
    include_robust = "multiscale_robust" in selected
    use_multiscale = include_lab or include_robust
    started = perf_counter()
    _validate_rgb_regions(rgb, regions)
    # Os extratores históricos convertem float16 com precisões diferentes.
    shared_lab = rgb2lab(rgb) if base_groups and use_multiscale and rgb.dtype != np.float16 else None
    if base_groups:
        ids, features = extract_region_features(
            rgb, regions, neighborhood_radius=neighborhood_radius, _groups=base_groups, _lab=shared_lab,
        )
    if timings is not None:
        timings["base"] = perf_counter() - started if base_groups else 0.0
    if use_multiscale:
        graph_timings = {} if timings is not None else None
        graph_ids, graph_features = extract_multiscale_context(
            rgb, regions, _include_lab=include_lab, _include_robust=include_robust, _lab=shared_lab,
            timings=graph_timings,
        )
        if timings is not None:
            timings.update(multiscale=graph_timings["total"], robust=graph_timings["robust"])
        if not base_groups:
            ids, features = graph_ids, graph_features
        elif np.array_equal(ids, graph_ids):
            features = np.column_stack((features, graph_features))
        else:
            raise ValueError("IDs das características originais e multiescala divergentes.")
    names = tuple(name for group in selected for name in FEATURE_GROUPS[group])
    return ids, features, names
