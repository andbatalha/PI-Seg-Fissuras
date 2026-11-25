"""Investiga os falsos positivos da baseline Stain nos IDs de validação."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np

from experiments.evaluate_regions import visualize_prediction
from experiments.reporting import guard_cli_output, write_summary
from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.dataset.masks import create_binary_mask
from src.evaluation import count_binary_pixels, pixel_metrics
from src.region_labeling import label_regions
from src.region_prediction import predict_regions


BASELINE_COUNTS = {"TP": 679462, "TN": 39652337, "FP": 2345849, "FN": 576112}
EXPECTED_REGIONS = 103621
I01 = "DJI_20250624181834_0007"
HIGH_PROBABILITY = 0.90
OCCUPANCY_BINS = ("0%", "(0%, 10%]", "(10%, 20%]", "(20%, 30%]", "(30%, 40%]",
                  "(40%, 50%]", "(50%, 60%]", "(60%, 70%)", "[70%, 80%]",
                  "(80%, 90%]", "(90%, 100%)", "100%")
PROBABILITY_BINS = ("[0.50, 0.60)", "[0.60, 0.70)", "[0.70, 0.80)", "[0.80, 0.90)", "[0.90, 1.00]")
NEGATIVE_CLASSES = tuple(damage for damage in DamageClass if damage != DamageClass.STAIN)


def ratio(numerator, denominator) -> float:
    """Mantém a convenção de avaliação do projeto para denominador nulo."""
    return float(numerator / denominator) if denominator else 0.0


def occupancy_bin(stain: int, area: int) -> str:
    """Usa contagem inteira para classificar exatamente os limites de ocupação."""
    if stain == 0:
        return OCCUPANCY_BINS[0]
    if stain == area:
        return OCCUPANCY_BINS[-1]
    percent_numerator = 100 * stain
    if percent_numerator < 70 * area:
        return OCCUPANCY_BINS[(10 * stain + area - 1) // area]
    if percent_numerator <= 80 * area:
        return OCCUPANCY_BINS[8]
    return OCCUPANCY_BINS[9 if percent_numerator <= 90 * area else 10]


def probability_bin(probability: float) -> str:
    """Agrupa previsões positivas com limites à esquerda inclusivos."""
    if not 0.5 <= probability <= 1.0:
        raise ValueError(f"Probabilidade fora dos intervalos de previsões positivas: {probability}")
    return PROBABILITY_BINS[int(np.searchsorted((0.60, 0.70, 0.80, 0.90), probability, side="right"))]


def class_pixel_counts(label: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conta o suporte das classes originais e seus pixels previstos como Stain."""
    if label.shape != mask.shape or not np.isin(label, [int(damage) for damage in DamageClass]).all():
        raise ValueError("Label BFDD incompatível com a predição ou com IDs de classe desconhecidos.")
    return np.bincount(label.ravel(), minlength=6), np.bincount(label[mask == 1], minlength=6)


def class_rows(scope: str, totals: np.ndarray, predicted: np.ndarray) -> list[dict]:
    fp = int(predicted[:int(DamageClass.STAIN)].sum())
    return [{"image_id": scope, "class_id": int(damage), "class": damage.name,
             "pixels": int(totals[damage]), "false_positives": int(predicted[damage]),
             "fp_share": ratio(predicted[damage], fp), "false_positive_rate": ratio(predicted[damage], totals[damage])}
            for damage in NEGATIVE_CLASSES]


def load_oracle(path: Path) -> dict:
    """Indexa os dados do D01 pelo par de IDs original."""
    regions = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["image_id"], int(row["region_id"])
            if key in regions:
                raise ValueError(f"Região duplicada no D01: {key}")
            regions[key] = (int(row["area"]), int(row["stain_pixels"]), int(row["non_stain_pixels"]),
                            float(row["stain_occupancy"]), int(row["max_iou_prediction"]))
    return regions


def region_rows(image_id: str, prediction, labeling, oracle: dict) -> list[dict]:
    """Alinha as regiões aos resultados do D01 e exporta também as previsões negativas."""
    if not np.array_equal(prediction.region_ids, labeling.region_ids):
        raise ValueError(f"A ordem dos IDs da inferência e da anotação diverge em {image_id}.")
    rows = []
    for position, region_id in enumerate(map(int, prediction.region_ids)):
        area, stain = int(labeling.pixel_counts[position]), int(labeling.target_counts[position])
        other = area - stain
        occupancy = float(labeling.target_fractions[position])
        key = image_id, region_id
        saved = oracle.pop(key, None)
        if saved is None or saved[:4] != (area, stain, other, occupancy) or saved[4] not in (0, 1):
            raise ValueError(f"Região ausente ou com estatísticas divergentes do D01: {key}")
        original = int(prediction.predictions[position])
        rows.append({"image_id": image_id, "region_id": region_id, "area": area,
                     "stain_pixels": stain, "non_stain_pixels": other, "stain_occupancy": occupancy,
                     "probability_stain": float(prediction.probabilities[position]),
                     "original_prediction": original, "training_label": int(labeling.labels[position]),
                     "occupancy_bin": occupancy_bin(stain, area), "oracle_max_iou_prediction": saved[4],
                     "TP": stain if original else 0, "FP": other if original else 0})
    return rows


def summarize_regions(rows: list[dict], global_fp: int) -> dict:
    selected = [row for row in rows if row["original_prediction"]]
    probabilities = [row["probability_stain"] for row in selected]
    tp, fp = (sum(row[key] for row in selected) for key in ("TP", "FP"))
    return {"regions": len(rows), "predicted_stain_regions": len(selected),
            "predicted_stain_share": ratio(len(selected), len(rows)), "TP": tp, "FP": fp,
            "fp_share": ratio(fp, global_fp), "pixel_precision": ratio(tp, tp + fp),
            "mean_probability": mean(probabilities) if selected else None,
            "median_probability": median(probabilities) if selected else None,
            "mean_occupancy": mean(row["stain_occupancy"] for row in selected) if selected else None}


def sum_counts(images: list[dict]) -> dict:
    return {key: sum(row[key] for row in images) for key in BASELINE_COUNTS}


def summarize_presence(images: list[dict], global_fp: int) -> dict:
    counts = sum_counts(images)
    false_positives = [row["FP"] for row in images]
    affected = sum(fp > 0 for fp in false_positives)
    return {"images": len(images), "pixels": sum(counts.values()), **counts,
            "images_with_fp": affected, "images_with_fp_share": ratio(affected, len(images)),
            "mean_fp_per_image": mean(false_positives) if images else 0.0,
            "median_fp_per_image": median(false_positives) if images else 0.0,
            "false_positive_rate": ratio(counts["FP"], counts["FP"] + counts["TN"]),
            "fp_share": ratio(counts["FP"], global_fp)}


def analyze_image(dataset: Path, image_id: str, artifact: dict, oracle: dict):
    """Executa a inferência original antes de abrir a máscara de referência."""
    started = perf_counter()
    rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
    prediction = predict_regions(artifact, rgb, include_details=True)
    inference_seconds = perf_counter() - started
    label = load_label(dataset / "Label" / f"{image_id}.png")
    reference = create_binary_mask(label, DamageClass.STAIN)
    counts = count_binary_pixels(reference, prediction.mask)
    labeling = label_regions(prediction.regions, label, **artifact["pipeline_config"]["labeling"])
    rows = region_rows(image_id, prediction, labeling, oracle)
    totals, predicted = class_pixel_counts(label, prediction.mask)
    if int(predicted[:5].sum()) != counts["FP"] or any(sum(r[k] for r in rows) != counts[k] for k in ("TP", "FP")):
        raise ValueError(f"Contagens de classe, região e máscara divergentes em {image_id}.")
    image = {"image_id": image_id, "has_stain": bool(reference.any()), "pixels": reference.size,
             "regions": len(rows), **counts, "false_positive_rate": ratio(counts["FP"], counts["TN"] + counts["FP"]),
             "fp_zero_occupancy": sum(r["FP"] for r in rows if r["stain_pixels"] == 0),
             "fp_high_probability": sum(r["FP"] for r in rows if r["probability_stain"] >= HIGH_PROBABILITY),
             "inference_seconds_including_rgb_load": inference_seconds}
    return image, rows, totals, predicted, (reference, prediction.mask)


def rankings(images: list[dict], global_fp: int) -> dict:
    def ranked(key, subset=images):
        return [dict(rank=index, image_id=row["image_id"], value=row[key], FP=row["FP"])
                for index, row in enumerate(sorted(subset, key=lambda row: (-row[key], row["image_id"])), start=1)]

    absolute = ranked("FP")
    return {"absolute_fp": absolute, "false_positive_rate": ranked("false_positive_rate"),
            "without_stain": ranked("FP", [row for row in images if not row["has_stain"]]),
            "zero_occupancy_fp": ranked("fp_zero_occupancy"), "high_probability_fp": ranked("fp_high_probability"),
            "top_concentration": {str(n): {"FP": sum(row["FP"] for row in absolute[:n]),
                                           "fp_share": ratio(sum(row["FP"] for row in absolute[:n]), global_fp)}
                                  for n in (5, 10, 20)}}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_charts(output: Path, classes: list, occupancy: list, probabilities: list,
                presence: dict, ranked: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plots = [
        ("fp_by_class", [row["class"] for row in classes], [row["false_positives"] for row in classes], "Pixels FP"),
        ("fp_rate_by_class", [row["class"] for row in classes], [100 * row["false_positive_rate"] for row in classes], "Taxa de FP (%)"),
        ("fp_by_occupancy", [row["group"] for row in occupancy], [row["FP"] for row in occupancy], "Pixels FP"),
        ("fp_by_probability", [row["group"] for row in probabilities], [row["FP"] for row in probabilities], "Pixels FP"),
        ("fp_by_presence", ["Com Stain", "Sem Stain"], [presence[key]["FP"] for key in ("with_stain", "without_stain")], "Pixels FP"),
    ]
    for name, labels, values, ylabel in plots:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.bar(labels, values, color="#356d91")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", labelrotation=35)
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=150)
        plt.close(figure)

    values = np.cumsum([0, *(row["FP"] for row in ranked["absolute_fp"])])
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(100 * values / max(int(values[-1]), 1), color="#356d91")
    axis.set(xlabel="Imagens em ordem decrescente de FP", ylabel="Participação acumulada (%)", ylim=(0, 100))
    figure.tight_layout()
    figure.savefig(output / "fp_concentration.png", dpi=150)
    plt.close(figure)


def save_examples(dataset: Path, output: Path, cache: dict, ranked: dict) -> dict:
    examples = {I01: ["investigação I01"]} if I01 in cache else {}
    for key in ("absolute_fp", "without_stain", "high_probability_fp"):
        for row in ranked[key][:2]:
            if row["value"] > 0:
                examples.setdefault(row["image_id"], []).append(f"top 2: {key}")
    for image_id in examples:
        rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
        visualize_prediction(rgb, *cache[image_id], output / "examples" / f"{image_id}.png")
    return examples


def run_experiment(dataset: Path, baseline: Path, d01: Path, output: Path) -> dict:
    started = perf_counter()
    manifest = json.loads((baseline / "split.json").read_text(encoding="utf-8"))
    ids = manifest["validation"]
    if len(ids) != 132 or len(set(ids)) != 132:
        raise ValueError("O manifesto deve fornecer 132 IDs distintos de validação.")
    artifact = joblib.load(baseline / "random_forest.joblib")
    config = artifact["pipeline_config"]
    d01_config = json.loads((d01 / "configuration.json").read_text(encoding="utf-8"))
    if (d01_config["validation_ids"] != ids or d01_config["slic"] != config["slic"]
            or config["labeling"]["class_id"] != DamageClass.STAIN):
        raise ValueError("Manifesto, classe ou configuração SLIC incompatível com o D01.")
    oracle = load_oracle(d01 / "region_statistics.csv")
    if len(oracle) != EXPECTED_REGIONS:
        raise ValueError("O CSV do D01 não tem os 103.621 superpixels esperados.")
    baseline_rows = json.loads((baseline / "pixel_validation_per_image.json").read_text(encoding="utf-8"))
    baseline_by_id = {row["image_id"]: row["counts"] for row in baseline_rows}
    if len(baseline_rows) != 132 or set(baseline_by_id) != set(ids):
        raise ValueError("O relatório da baseline não corresponde ao manifesto de validação.")

    images, regions, classes = [], [], []
    cache = {}
    all_totals, all_predicted = np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)
    for index, image_id in enumerate(ids, start=1):
        image, rows, totals, predicted, masks = analyze_image(dataset, image_id, artifact, oracle)
        if {key: image[key] for key in BASELINE_COUNTS} != baseline_by_id[image_id]:
            raise ValueError(f"Divergência em {image_id}: atual {sum_counts([image])}, baseline {baseline_by_id[image_id]}")
        images.append(image)
        regions.extend(rows)
        classes.extend(class_rows(image_id, totals, predicted))
        all_totals += totals
        all_predicted += predicted
        cache[image_id] = masks
        if index == 1 or index % 20 == 0 or index == len(ids):
            print(f"Validação D02: {index}/{len(ids)}", flush=True)
    if oracle or len(regions) != EXPECTED_REGIONS or sum_counts(images) != BASELINE_COUNTS:
        raise ValueError("As contagens globais ou regiões não coincidem com a baseline e o D01.")

    global_fp = BASELINE_COUNTS["FP"]
    positive = [row for row in regions if row["original_prediction"]]
    occupancy = [{"group": group, **summarize_regions([r for r in regions if r["occupancy_bin"] == group], global_fp)}
                 for group in OCCUPANCY_BINS]
    training = [{"group": name, **summarize_regions([r for r in regions if r["training_label"] == code], global_fp)}
                for code, name in ((0, "negative"), (-1, "ambiguous"), (1, "positive"))]
    for row in positive:
        row["probability_bin"] = probability_bin(row["probability_stain"])
    probabilities = [{"group": group, **summarize_regions([r for r in positive if r["probability_bin"] == group], global_fp)}
                     for group in PROBABILITY_BINS]
    for row in regions:
        row.setdefault("probability_bin", "")
    presence = {name: summarize_presence([row for row in images if row["has_stain"] is flag], global_fp)
                for flag, name in ((True, "with_stain"), (False, "without_stain"))}
    if (presence["with_stain"]["images"], presence["without_stain"]["images"]) != (75, 57):
        raise ValueError("A divisão por presença de Stain deve ser 75/57.")
    oracle_impact = {}
    for selected, name in ((1, "also_selected"), (0, "not_selected")):
        fp = sum(r["FP"] for r in positive if r["oracle_max_iou_prediction"] == selected)
        oracle_impact[name] = {"FP": fp, "fp_share": ratio(fp, global_fp)}
    high_negative = summarize_regions([r for r in positive if r["stain_pixels"] == 0
                                       and r["probability_stain"] >= HIGH_PROBABILITY], global_fp)
    overall_classes = class_rows("GLOBAL", all_totals, all_predicted)
    for group in (overall_classes, occupancy, probabilities, training, presence.values(), oracle_impact.values()):
        if sum(row["false_positives"] if "false_positives" in row else row["FP"] for row in group) != global_fp:
            raise ValueError("Uma partição de falsos positivos não reproduziu o total da baseline.")

    ranked = rankings(images, global_fp)
    output.mkdir(parents=True, exist_ok=True)
    examples = save_examples(dataset, output, cache, ranked)
    save_charts(output / "charts", overall_classes, occupancy, probabilities, presence, ranked)
    write_csv(output / "false_positives_by_class.csv", overall_classes + classes)
    write_csv(output / "false_positives_by_image.csv", images)
    write_csv(output / "false_positives_by_occupancy.csv", occupancy)
    write_csv(output / "false_positives_by_probability.csv", probabilities)
    write_csv(output / "region_statistics.csv", regions)
    write_json(output / "false_positives_by_presence.json", presence)
    write_json(output / "false_positives_by_oracle.json", oracle_impact)
    write_json(output / "image_rankings.json", ranked)
    write_json(output / "configuration.json", {"validation_ids": ids, "manifest": (baseline / "split.json").as_posix(),
               "model": (baseline / "random_forest.joblib").as_posix(), "pipeline_config": config,
               "feature_names": artifact["feature_names"], "d01_region_statistics": (d01 / "region_statistics.csv").as_posix(),
               "decision": "predict()", "high_probability_min": HIGH_PROBABILITY, "dataset": dataset.as_posix()})
    report = {"images": len(images), "pixels": sum(BASELINE_COUNTS.values()), "counts": BASELINE_COUNTS,
              "metrics": pixel_metrics(BASELINE_COUNTS), "regions": len(regions), "predicted_stain_regions": len(positive),
              "by_training_label": training, "high_probability_zero_occupancy": high_negative,
              "examples": examples, "inference_seconds_including_rgb_load": sum(r["inference_seconds_including_rgb_load"] for r in images),
              "wall_seconds": perf_counter() - started}
    write_json(output / "global_statistics.json", report)
    write_summary(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--baseline", type=Path, default=Path("results/region_full"))
    parser.add_argument("--d01", type=Path, default=Path("results/diagnostics/D01_slic_oracle"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/D02_false_positives"))
    args = parser.parse_args()
    targets = tuple(path.name for path in args.output.iterdir()) if args.output.is_dir() else ()
    guard_cli_output(args.output, targets, owner="D02_false_positives")
    print(json.dumps(run_experiment(args.dataset, args.baseline, args.d01, args.output), indent=2))


if __name__ == "__main__":
    main()
