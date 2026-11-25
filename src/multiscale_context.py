"""Descritores de contexto em distâncias de grafo entre superpixels."""

from time import perf_counter

import numpy as np
from skimage.color import rgb2lab


MULTISCALE_NAMES = (
    "delta_l_r2", "delta_a_r2", "delta_b_r2", "delta_l_r4", "delta_a_r4", "delta_b_r4",
    "delta_l_p75_r2", "delta_l_p75_r4", "context_l_iqr_r2", "context_l_iqr_r4",
)


def region_adjacency(regions: np.ndarray) -> dict[int, set[int]]:
    """Cria um grafo não direcionado usando fronteiras horizontais e verticais."""
    graph = {int(region_id): set() for region_id in np.unique(regions)}
    boundaries = ((regions[:, :-1], regions[:, 1:]), (regions[:-1], regions[1:]))
    for first, second in boundaries:
        different = first != second
        pairs = np.column_stack((first[different], second[different]))
        for left, right in np.unique(np.sort(pairs, axis=1), axis=0).tolist():
            graph[left].add(right)
            graph[right].add(left)
    return graph


def region_neighborhoods(graph: dict[int, set[int]], region_id: int) -> tuple[set[int], ...]:
    """Retorna vizinhanças cumulativas N1, N2 e N4, sem o próprio ID."""
    visited, frontier = {region_id}, {region_id}
    neighborhoods = []
    for distance in range(1, 5):
        frontier = set().union(*(graph[node] for node in frontier)) - visited
        visited.update(frontier)
        if distance in (1, 2, 4):
            neighborhoods.append(visited - {region_id})
    return tuple(neighborhoods)


def extract_multiscale_context(rgb: np.ndarray, regions: np.ndarray, *,
                               timings: dict | None = None,
                               _include_lab: bool = True,
                               _include_robust: bool = True,
                               _lab: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extrai dez colunas do RGB e do mapa SLIC. Vizinhanças vazias retornam zero."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or regions.shape != rgb.shape[:2] or not regions.size:
        raise ValueError("RGB e mapa SLIC devem ter dimensões espaciais iguais e não vazias.")
    started = perf_counter()
    robust_seconds = 0.0
    lab = (_lab if _lab is not None else rgb2lab(rgb)).reshape(-1, 3)
    region_ids, inverse, counts = np.unique(regions.ravel(), return_inverse=True, return_counts=True)
    rows = {int(region_id): i for i, region_id in enumerate(region_ids)}
    channels = lab.T if _include_lab else lab.T[:1]
    sums = np.column_stack([np.bincount(inverse, weights=channel) for channel in channels])
    means = sums / counts[:, None]
    if _include_robust:
        robust_started = perf_counter()
        pixels = np.split(lab[np.argsort(inverse, kind="stable"), 0], np.cumsum(counts)[:-1])
        robust_seconds += perf_counter() - robust_started
    graph = region_adjacency(regions)
    features = np.zeros((len(region_ids), len(MULTISCALE_NAMES)))
    for i, region_id in enumerate(region_ids):
        _, radius_two, radius_four = region_neighborhoods(graph, int(region_id))
        for scale, neighbors in enumerate((radius_two, radius_four)):
            if not neighbors:
                continue
            indices = [rows[node] for node in sorted(neighbors)]
            if _include_lab:
                first = 3 * scale
                features[i, first:first + 3] = means[i] - sums[indices].sum(axis=0) / counts[indices].sum()
            if _include_robust:
                robust_started = perf_counter()
                lightness = np.concatenate([pixels[index] for index in indices])
                lower, upper = np.percentile(lightness, [25, 75], overwrite_input=True)
                features[i, 6 + scale] = means[i, 0] - upper
                features[i, 8 + scale] = upper - lower
                robust_seconds += perf_counter() - robust_started
    if not np.isfinite(features).all():
        raise ValueError("Os descritores multiescala contêm valores não finitos.")
    if timings is not None:
        timings.update(total=perf_counter() - started, robust=robust_seconds)
    if not _include_lab:
        features = features[:, 6:]
    elif not _include_robust:
        features = features[:, :6]
    return region_ids, features
