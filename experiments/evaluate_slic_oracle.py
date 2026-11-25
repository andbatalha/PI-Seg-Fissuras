"""Avalia o limite de segmentação binária imposto pelos superpixels do SLIC."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

from experiments.reporting import guard_cli_output, write_summary
from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.dataset.masks import create_binary_mask
from src.evaluation import count_binary_pixels, pixel_metrics
from src.region_labeling import label_regions
from src.region_prediction import reconstruct_binary_mask
from src.segmentation import segment_regions
from visualization.plotting import comparison_panels, pixel_error_legend, pixel_error_map, save_comparison


VARIANTS = ("majority", "max_iou")
I01_COUNTS = {"TP": 3044, "TN": 316025, "FP": 3686, "FN": 4925}


@dataclass(frozen=True)
class ImageRegions:
    image_id: str
    regions: np.ndarray
    region_ids: np.ndarray
    reference: np.ndarray
    areas: np.ndarray
    stain_pixels: np.ndarray
    preparation_seconds: float


def majority_selection(areas: np.ndarray, stain_pixels: np.ndarray) -> np.ndarray:
    """Seleciona regiões com maioria estrita de pixels Stain."""
    return stain_pixels > areas - stain_pixels


def max_iou_selection(areas: np.ndarray, stain_pixels: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Maximiza a IoU global e devolve a decisão e a menor ocupação selecionada."""
    selected = np.zeros(areas.size, dtype=bool)
    total_stain = int(stain_pixels.sum())
    if total_stain == 0:
        return selected, None

    occupancy = stain_pixels / areas
    order = np.argsort(-occupancy, kind="stable")
    true_positives = np.cumsum(stain_pixels[order], dtype=np.int64)
    false_positives = np.cumsum((areas - stain_pixels)[order], dtype=np.int64)

    # A união vale o total real de Stain mais os falsos positivos do prefixo.
    # Produtos inteiros evitam desempates por arredondamento das IoUs.
    best_cut = best_tp = 0
    best_union = total_stain
    for cut, (tp, fp) in enumerate(zip(true_positives.tolist(), false_positives.tolist()), start=1):
        union = total_stain + fp
        if tp * best_union > best_tp * union:
            best_cut, best_tp, best_union = cut, tp, union

    selected[order[:best_cut]] = True
    return selected, float(occupancy[order[best_cut - 1]])


def collect_image(dataset: Path, image_id: str, config: dict) -> ImageRegions:
    """Calcula SLIC e ocupação real para um único ID previamente autorizado."""
    started = perf_counter()
    rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
    label = load_label(dataset / "Label" / f"{image_id}.png")
    regions = segment_regions(rgb, **config["slic"]).astype(np.int32, copy=False)
    labeled = label_regions(regions, label, **config["labeling"])
    reference = create_binary_mask(label, DamageClass.STAIN)
    return ImageRegions(image_id, regions, labeled.region_ids, reference,
                        labeled.pixel_counts, labeled.target_counts, perf_counter() - started)


def evaluate_selection(image: ImageRegions, selected: np.ndarray) -> tuple[dict, np.ndarray]:
    """Reconstrói a máscara por ID e usa as métricas binárias do projeto."""
    started = perf_counter()
    mask = reconstruct_binary_mask(image.regions, image.region_ids, selected)
    counts = count_binary_pixels(image.reference, mask)
    report = {"counts": counts, "metrics": pixel_metrics(counts),
              "evaluation_seconds": perf_counter() - started}
    return report, mask


def summarize_occupancy(areas: np.ndarray, stain_pixels: np.ndarray) -> dict:
    """Agrupa ocupações em [0,10), …, [90,100], com valores em porcentagem."""
    bins = np.minimum((10 * stain_pixels) // areas, 9)
    histogram = np.bincount(bins, minlength=10)
    return {
        "total_regions": int(areas.size),
        "without_stain": int(np.count_nonzero(stain_pixels == 0)),
        "fully_stain": int(np.count_nonzero(stain_pixels == areas)),
        "partially_stain": int(np.count_nonzero((stain_pixels > 0) & (stain_pixels < areas))),
        "bins": [
            {"min_percent": i * 10, "max_percent": (i + 1) * 10, "max_inclusive": i == 9,
             "regions": int(count)} for i, count in enumerate(histogram)
        ],
    }


def example_indices(records: list[dict], baseline_rows: dict) -> list[int]:
    """Escolhe I01 por correspondência única e amostras baixo, mediano, alto e sem Stain."""
    i01 = [i for i, row in enumerate(records)
           if baseline_rows[row["image_id"]]["counts"] == I01_COUNTS]
    positive = sorted((i for i, row in enumerate(records) if row["has_stain"]),
                      key=lambda i: records[i]["max_iou"]["metrics"]["IoU"])
    negative = next((i for i, row in enumerate(records) if not row["has_stain"]), None)
    choices = i01 if len(i01) == 1 else []
    if positive:
        choices.extend((positive[0], positive[len(positive) // 2], positive[-1]))
    if negative is not None:
        choices.append(negative)
    return list(dict.fromkeys(choices))


def save_example(image: ImageRegions, selections: dict, dataset: Path, output: Path) -> None:
    """Grava RGB, referência, fronteiras e os dois pares de máscara e erro."""
    rgb = load_image(dataset / "RGB" / f"{image.image_id}.JPG")
    panels = [(rgb, "RGB"), (image.reference, "Ground Truth"),
              (mark_boundaries(rgb, image.regions), "Fronteiras SLIC")]
    for variant, title in (("majority", "Majority"), ("max_iou", "Max-IoU")):
        mask = reconstruct_binary_mask(image.regions, image.region_ids, selections[variant])
        panels.extend([(mask, title), (pixel_error_map(image.reference, mask), f"Erros · {title}")])
    figure, axes = comparison_panels(panels, columns=4)
    axes[-1].legend(handles=pixel_error_legend(), loc="upper center",
                   bbox_to_anchor=(0.5, 0), ncol=2, fontsize=8)
    save_comparison(figure, output)
    plt.close(figure)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_experiment(dataset: Path, baseline: Path, output: Path) -> dict:
    """Executa uma única segmentação SLIC por imagem e avalia dois oráculos."""
    started = perf_counter()
    manifest_path = baseline / "split.json"
    model_path = baseline / "random_forest.joblib"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = manifest["validation"]
    if len(ids) != 132 or len(set(ids)) != 132:
        raise ValueError("O manifesto deve conter exatamente 132 IDs distintos de validação.")
    config = joblib.load(model_path)["pipeline_config"]
    if config["labeling"]["class_id"] != DamageClass.STAIN:
        raise ValueError("A configuração da baseline precisa usar Stain como classe-alvo.")
    baseline_report = json.loads((baseline / "pixel_validation_metrics.json").read_text(encoding="utf-8"))
    rows = json.loads((baseline / "pixel_validation_per_image.json").read_text(encoding="utf-8"))
    baseline_rows = {row["image_id"]: row for row in rows}
    if len(rows) != len(ids) or set(baseline_rows) != set(ids) or baseline_report["images"] != len(ids):
        raise ValueError("Os relatórios da baseline não correspondem aos IDs de validação.")

    images = []
    for index, image_id in enumerate(ids, start=1):
        images.append(collect_image(dataset, image_id, config))
        if index == 1 or index % 20 == 0 or index == len(ids):
            print(f"SLIC e ocupação: {index}/{len(ids)}", flush=True)
    areas = np.concatenate([image.areas for image in images])
    stain_pixels = np.concatenate([image.stain_pixels for image in images])
    if (int(areas.sum()) != baseline_report["pixels"] or int(stain_pixels.sum()) !=
            baseline_report["counts"]["TP"] + baseline_report["counts"]["FN"]):
        raise ValueError("O total de pixels ou o suporte de Stain diverge da baseline.")

    selection_seconds = {}
    selection_started = perf_counter()
    majority = majority_selection(areas, stain_pixels)
    selection_seconds["majority"] = perf_counter() - selection_started
    selection_started = perf_counter()
    optimal, cutoff = max_iou_selection(areas, stain_pixels)
    selection_seconds["max_iou"] = perf_counter() - selection_started
    boundaries = np.cumsum([image.region_ids.size for image in images])[:-1]
    selections = [{"majority": left, "max_iou": right} for left, right in
                  zip(np.split(majority, boundaries), np.split(optimal, boundaries))]

    records = []
    for image, decisions in zip(images, selections):
        expected = baseline_rows[image.image_id]["counts"]
        if image.stain_pixels.sum() != expected["TP"] + expected["FN"]:
            raise ValueError(f"O suporte de Stain de {image.image_id} diverge da baseline.")
        row = {"image_id": image.image_id, "pixels": image.reference.size,
               "has_stain": bool(image.reference.any()), "superpixels": image.region_ids.size,
               "preparation_seconds": image.preparation_seconds}
        for variant in VARIANTS:
            row[variant], _ = evaluate_selection(image, decisions[variant])
            row[variant]["processing_seconds"] = (
                image.preparation_seconds + row[variant]["evaluation_seconds"]
            )
        records.append(row)

    occupancy = summarize_occupancy(areas, stain_pixels)
    report = {"images": len(images), "pixels": int(areas.sum()),
              "images_without_stain": sum(not row["has_stain"] for row in records),
              "baseline": {"counts": baseline_report["counts"], "metrics": baseline_report["metrics"],
                           "inference_seconds": baseline_report["inference_seconds"]},
              "max_iou_occupancy_cutoff": cutoff, "occupancy": occupancy}
    preparation = sum(image.preparation_seconds for image in images)
    report["preparation_seconds"] = preparation
    for variant, selected in (("majority", majority), ("max_iou", optimal)):
        counts = {name: sum(row[variant]["counts"][name] for row in records)
                  for name in ("TP", "TN", "FP", "FN")}
        evaluation = sum(row[variant]["evaluation_seconds"] for row in records)
        report[variant] = {"counts": counts, "metrics": pixel_metrics(counts),
                           "selected_regions": int(selected.sum()), "evaluation_seconds": evaluation,
                           "selection_seconds": selection_seconds[variant],
                           "processing_seconds": preparation + selection_seconds[variant] + evaluation}
    best_iou = report["max_iou"]["metrics"]["IoU"]
    if best_iou < report["majority"]["metrics"]["IoU"]:
        raise RuntimeError("A IoU do oráculo global ficou abaixo da seleção por maioria.")
    report["baseline_to_max_iou_gap"] = best_iou - baseline_report["metrics"]["IoU"]

    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for index in example_indices(records, baseline_rows):
        image = images[index]
        filename = Path("examples") / f"{image.image_id}.png"
        save_example(image, selections[index], dataset, output / filename)
        examples.append({"image_id": image.image_id, "file": filename.as_posix(),
                         "has_stain": records[index]["has_stain"],
                         "matches_i01_counts": baseline_rows[image.image_id]["counts"] == I01_COUNTS})
    report["examples"] = examples
    write_json(output / "configuration.json", {
        "experiment": "D01_slic_oracle", "target_class_id": int(DamageClass.STAIN),
        "manifest": manifest_path.as_posix(), "baseline_model": model_path.as_posix(),
        "validation_ids": ids, "slic": config["slic"], "dataset": dataset.as_posix(),
        "majority_rule": "occupancy > 0.5",
        "max_iou_rule": "global descending occupancy prefix; ties in IoU favor the shortest prefix",
    })
    write_json(output / "per_image_metrics.json", records)
    write_json(output / "occupancy_distribution.json", occupancy)
    with (output / "region_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "region_id", "area", "stain_pixels", "non_stain_pixels",
                         "stain_occupancy", "majority_prediction", "max_iou_prediction"])
        for image, decisions in zip(images, selections):
            writer.writerows(zip(
                [image.image_id] * image.region_ids.size, image.region_ids, image.areas,
                image.stain_pixels, image.areas - image.stain_pixels, image.stain_pixels / image.areas,
                decisions["majority"].astype(np.uint8), decisions["max_iou"].astype(np.uint8),
            ))
    report["wall_seconds"] = perf_counter() - started
    write_json(output / "global_metrics.json", report)
    write_summary(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--baseline", type=Path, default=Path("results/region_full"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/D01_slic_oracle"))
    args = parser.parse_args()
    targets = tuple(path.name for path in args.output.iterdir()) if args.output.is_dir() else ()
    guard_cli_output(args.output, targets, owner="D01_slic_oracle")
    print(json.dumps(run_experiment(args.dataset, args.baseline, args.output), indent=2))


if __name__ == "__main__":
    main()
