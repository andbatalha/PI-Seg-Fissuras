import json
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import numpy as np

from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.feature_extractor import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS, canonical_feature_groups, extract_features
from src.region_labeling import label_regions
from src.segmentation import segment_regions


PIPELINE_CONFIG = {
    "slic": {"n_segments": 800, "compactness": 10, "sigma": 1, "start_label": 1},
    "labeling": {"class_id": 5, "negative_max": 0.10, "positive_min": 0.70},
    "features": {"neighborhood_radius": 5, "quantization_levels": 32},
}


def build_image_rows(dataset_root: Path, image_id: str, *, feature_groups: Iterable[str] = DEFAULT_FEATURE_GROUPS,
                     pipeline_config: dict | None = None, include_class_counts: bool = False,
                     timings: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """Extrai características e metadados de todas as regiões de uma imagem."""
    config = PIPELINE_CONFIG if pipeline_config is None else pipeline_config
    started = perf_counter()
    rgb = load_image(dataset_root / "RGB" / f"{image_id}.JPG")
    regions = segment_regions(rgb, **config["slic"])
    if timings is not None:
        timings["rgb_slic"] = perf_counter() - started
    region_ids, features, _ = extract_features(
        rgb, regions, feature_groups, neighborhood_radius=config["features"]["neighborhood_radius"],
        timings=timings,
    )
    ground_truth = load_label(dataset_root / "Label" / f"{image_id}.png")
    labeling = label_regions(regions, ground_truth, **config["labeling"])
    if not np.array_equal(region_ids, labeling.region_ids):
        raise ValueError("Os IDs das características e dos rótulos divergem.")
    if not np.isfinite(features).all():
        raise ValueError("A matriz de características contém valores inválidos.")

    rows = {
        "region_ids": region_ids,
        "features": features,
        "pixel_counts": labeling.pixel_counts,
        "target_counts": labeling.target_counts,
        "target_fractions": labeling.target_fractions,
        "labels": labeling.labels,
    }
    if include_class_counts:
        bins = labeling.pixel_indices.ravel() * len(DamageClass) + ground_truth.ravel()
        rows["class_pixel_counts"] = np.bincount(bins, minlength=len(region_ids) * len(DamageClass)).reshape(
            -1, len(DamageClass),
        )
    return rows


def save_region_dataset(output: Path, data: dict, feature_groups: Iterable[str]) -> None:
    """Grava o NPZ com nomes e grupos canônicos, preservando os demais campos."""
    groups = canonical_feature_groups(feature_groups)
    names = tuple(name for group in groups for name in FEATURE_GROUPS[group])
    if data["features"].ndim != 2 or data["features"].shape[1] != len(names):
        raise ValueError("A quantidade de colunas não corresponde aos grupos selecionados.")
    np.savez_compressed(output, **{**data, "feature_names": names, "feature_groups": groups})


def build_region_dataset(dataset_root: Path, image_ids: list[str], output: Path, *,
                         feature_groups: Iterable[str] = DEFAULT_FEATURE_GROUPS) -> dict:
    """Grava um NPZ por subconjunto. Se uma imagem falhar, aborta a gravação."""
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Informe uma lista não vazia de imagens sem IDs duplicados.")
    groups = canonical_feature_groups(feature_groups)

    started = perf_counter()
    batches = []
    for index, image_id in enumerate(image_ids, start=1):
        try:
            batches.append(build_image_rows(dataset_root, image_id, feature_groups=groups))
        except Exception as error:
            raise RuntimeError(f"1 imagem com erro ({image_id}): {error}") from error
        if index == 1 or index % 25 == 0 or index == len(image_ids):
            print(f"{output.stem} | {index}/{len(image_ids)} imagens", flush=True)

    data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    # Cada linha referencia o ID da imagem pela posição na lista persistida.
    image_row_counts = [batch["labels"].size for batch in batches]
    data["image_indices"] = np.repeat(np.arange(len(image_ids)), image_row_counts)
    data.update(image_ids=image_ids, pipeline_config=json.dumps(PIPELINE_CONFIG))
    save_region_dataset(output, data, groups)
    elapsed = perf_counter() - started
    target_pixels = data["target_counts"].sum()
    categories = {}
    for name, label in (("positive", 1), ("negative", 0), ("ambiguous", -1)):
        selected = data["labels"] == label
        categories[name] = {
            "regions": int(selected.sum()),
            "percent_regions": float(100 * selected.mean()),
            "percent_target_pixels": (
                float(100 * data["target_counts"][selected].sum() / target_pixels)
                if target_pixels else None
            ),
        }
    images_with_target = sum(bool(batch["target_counts"].any()) for batch in batches)
    positives = categories["positive"]["regions"]
    return {
        "images": len(image_ids),
        "images_with_target": images_with_target,
        "images_without_target": len(image_ids) - images_with_target,
        "regions": data["labels"].size,
        "categories": categories,
        "negative_to_positive": categories["negative"]["regions"] / positives if positives else None,
        "feature_shape": data["features"].shape,
        "processing_seconds": elapsed,
        "seconds_per_image": elapsed / len(image_ids),
        "file_bytes": output.stat().st_size,
        "image_errors": 0,
    }
