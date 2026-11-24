import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.dataset.loader import load_image, load_label
from src.region_features import FEATURE_NAMES, extract_region_features
from src.region_labeling import label_regions
from src.segmentation import segment_regions


PIPELINE_CONFIG = {
    "slic": {"n_segments": 800, "compactness": 10, "sigma": 1, "start_label": 1},
    "labeling": {"class_id": 5, "negative_max": 0.10, "positive_min": 0.70},
    "features": {"neighborhood_radius": 5, "quantization_levels": 32},
}


def build_image_rows(dataset_root: Path, image_id: str) -> dict[str, np.ndarray]:
    """Extrai características e metadados de todas as regiões de uma imagem."""
    rgb = load_image(dataset_root / "RGB" / f"{image_id}.JPG")
    ground_truth = load_label(dataset_root / "Label" / f"{image_id}.png")
    regions = segment_regions(rgb, **PIPELINE_CONFIG["slic"])
    region_ids, features = extract_region_features(
        rgb, regions, neighborhood_radius=PIPELINE_CONFIG["features"]["neighborhood_radius"]
    )
    labeling = label_regions(regions, ground_truth, **PIPELINE_CONFIG["labeling"])
    if not np.array_equal(region_ids, labeling.region_ids):
        raise ValueError("Os IDs das características e dos rótulos divergem.")
    if not np.isfinite(features).all():
        raise ValueError("A matriz de características contém valores inválidos.")

    return {
        "region_ids": region_ids,
        "features": features,
        "pixel_counts": labeling.pixel_counts,
        "target_counts": labeling.target_counts,
        "target_fractions": labeling.target_fractions,
        "labels": labeling.labels,
    }


def build_region_dataset(dataset_root: Path, image_ids: list[str], output: Path) -> dict:
    """Grava um NPZ por subconjunto. Se uma imagem falhar, aborta a gravação."""
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Informe uma lista não vazia de imagens sem IDs duplicados.")

    started = perf_counter()
    batches = []
    for index, image_id in enumerate(image_ids, start=1):
        try:
            batches.append(build_image_rows(dataset_root, image_id))
        except Exception as error:
            raise RuntimeError(f"1 imagem com erro ({image_id}): {error}") from error
        if index == 1 or index % 25 == 0 or index == len(image_ids):
            print(f"{output.stem} | {index}/{len(image_ids)} imagens", flush=True)

    data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    # Cada linha referencia o ID da imagem pela posição na lista persistida.
    image_row_counts = [batch["labels"].size for batch in batches]
    data["image_indices"] = np.repeat(np.arange(len(image_ids)), image_row_counts)
    np.savez_compressed(
        output, **data, image_ids=image_ids, feature_names=FEATURE_NAMES,
        pipeline_config=json.dumps(PIPELINE_CONFIG),
    )
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
