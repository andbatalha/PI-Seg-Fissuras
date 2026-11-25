"""Avalia limiares de Stain usando as probabilidades exportadas no D02."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import guard_cli_output, write_summary
from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.dataset.masks import create_binary_mask
from src.evaluation import count_binary_pixels, count_region_pixels, pixel_metrics
from src.region_labeling import label_regions
from src.region_prediction import reconstruct_binary_mask
from src.segmentation import segment_regions
from visualization.plotting import comparison_panels, pixel_error_legend, pixel_error_map, save_comparison


BASELINE_COUNTS = {"TP": 679462, "TN": 39652337, "FP": 2345849, "FN": 576112}
I01 = "DJI_20250624181834_0007"
I01_REGIONS = {353: "TN", 356: "TN", 352: "FP", 127: "TP", 158: "TP", 764: "FN"}
REGION_DTYPE = [("image", "U40"), ("id", "i8"), ("area", "i8"), ("stain", "i8"),
                ("other", "i8"), ("probability", "f8"), ("original", "?")]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_regions(path: Path) -> np.ndarray:
    """Carrega do D02 somente as colunas necessárias à reavaliação."""
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        result = np.fromiter((
            (row["image_id"], int(row["region_id"]), int(row["area"]), int(row["stain_pixels"]),
             int(row["non_stain_pixels"]), float(row["probability_stain"]),
             {"0": False, "1": True}[row["original_prediction"]]) for row in rows
        ), dtype=REGION_DTYPE)
    if (not ((result["probability"] >= 0) & (result["probability"] <= 1)).all()
            or (result["area"] <= 0).any() or (result["stain"] < 0).any()
            or (result["other"] < 0).any()
            or not np.array_equal(result["area"], result["stain"] + result["other"])):
        raise ValueError("Contagens ou probabilidades inválidas no CSV do D02.")
    return result


def apply_threshold(probabilities: np.ndarray, threshold: float | None) -> np.ndarray:
    """Aplica P ≥ limiar. None representa a solução totalmente negativa."""
    return np.zeros(probabilities.shape, dtype=bool) if threshold is None else probabilities >= threshold


def flow_counts(regions: np.ndarray, selected: np.ndarray) -> dict:
    """Soma as contribuições por região, ponderadas pela quantidade de pixels."""
    return count_region_pixels(regions["stain"], regions["other"], selected)


def threshold_curve(regions: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Encontra prefixos por busca binária, incluindo por inteiro as probabilidades empatadas."""
    order = np.argsort(-regions["probability"], kind="stable")
    ordered = regions[order]
    selected = np.searchsorted(-ordered["probability"], -thresholds, side="right")
    tp = np.r_[0, np.cumsum(ordered["stain"])][selected]
    fp = np.r_[0, np.cumsum(ordered["other"])][selected]
    return {"TP": tp, "TN": regions["other"].sum() - fp, "FP": fp,
            "FN": regions["stain"].sum() - tp, "positive_regions": selected}


def exact_thresholds(regions: np.ndarray) -> np.ndarray:
    """Enumera os cortes distintos, do maior para o menor, incluindo tudo negativo."""
    return np.r_[np.inf, np.unique(regions["probability"])[::-1]]


def summarize_counts(counts: dict, positive_regions: int, empty_positive_images: int) -> dict:
    metrics = pixel_metrics(counts)
    negatives = counts["FP"] + counts["TN"]
    metrics["Specificity"] = counts["TN"] / negatives if negatives else 0.0
    metrics["false_positive_rate"] = counts["FP"] / negatives if negatives else 0.0
    return {**counts, **metrics, "positive_regions": int(positive_regions),
            "positive_pixels": counts["TP"] + counts["FP"],
            "without_stain_positive_images": int(empty_positive_images)}


def curve_row(curve: dict, position: int, empty_positive_images: int) -> dict:
    counts = {key: int(curve[key][position]) for key in BASELINE_COUNTS}
    return summarize_counts(counts, curve["positive_regions"][position], empty_positive_images)


def best_index(curve: dict, feasible: np.ndarray, *, f1: bool = False) -> int | None:
    """Compara razões exatas de inteiros. A ordem dos índices resolve empates."""
    best = None
    best_numerator, best_denominator = -1, 1
    for position in np.flatnonzero(feasible):
        tp, fp, fn = (int(curve[key][position]) for key in ("TP", "FP", "FN"))
        numerator = 2 * tp if f1 else tp
        denominator = numerator + fp + fn or 1
        if numerator * best_denominator > best_numerator * denominator:
            best = int(position)
            best_numerator, best_denominator = numerator, denominator
    return best


def optimal_threshold(regions: np.ndarray) -> tuple[float | None, np.ndarray, dict]:
    """Maximiza a IoU global e resolve empates pelo maior corte."""
    thresholds = exact_thresholds(regions)
    curve = threshold_curve(regions, thresholds)
    index = best_index(curve, np.ones(len(thresholds), dtype=bool))
    threshold = None if np.isinf(thresholds[index]) else float(thresholds[index])
    return threshold, thresholds, curve


def choose_optima(curve: dict, baseline: dict) -> dict:
    """Seleciona sobre todos os cortes, com restrições calculadas por contagens inteiras."""
    tp, fp, fn = (curve[key] for key in ("TP", "FP", "FN"))
    all_points = np.ones(tp.size, dtype=bool)
    recall_minimum = tp * (baseline["TP"] + baseline["FN"]) >= baseline["TP"] * (tp + fn)
    constraints = {"max_iou": all_points, "max_f1": all_points,
                   "recall_at_least_baseline": recall_minimum}
    for percent in (30, 40, 50):
        constraints[f"precision_at_least_{percent}"] = ((tp + fp) > 0) & (100 * tp >= percent * (tp + fp))
    optima = {key: best_index(curve, mask, f1=key == "max_f1") for key, mask in constraints.items()}
    if optima["max_iou"] != optima["max_f1"]:
        raise RuntimeError("IoU e F1 não selecionaram o mesmo corte com o mesmo desempate.")
    return optima


def export_points(path: Path, groups: dict, negative_maxima: np.ndarray, grid: np.ndarray,
                  exact: np.ndarray, baseline: dict) -> list[dict]:
    """Grava todos os pontos e devolve a série de 0,01 usada nos gráficos."""
    def rows():
        yield {"source": "baseline_predict", "threshold": "", **baseline}
        for source, thresholds in (("grid", grid), ("exact", exact)):
            curves = {name: threshold_curve(group, thresholds) for name, group in groups.items()}
            affected = negative_maxima.size - np.searchsorted(negative_maxima, thresholds, side="left")
            for index, threshold in enumerate(thresholds):
                report = {}
                for name, curve in curves.items():
                    count = affected[index] if name != "with_stain" else 0
                    prefix = f"{name}_" if name != "global" else ""
                    report.update((prefix + key, value) for key, value in curve_row(curve, index, count).items())
                yield {"source": "all_negative" if np.isinf(threshold) else source,
                       "threshold": "" if np.isinf(threshold) else float(threshold), **report}

    grid_rows = []
    with path.open("w", newline="", encoding="utf-8") as stream:
        iterator = rows()
        first = next(iterator)
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        for row in iterator:
            writer.writerow(row)
            if row["source"] == "grid":
                grid_rows.append(row)
    return grid_rows


def baseline_groups(grouped: dict, zero_positive_images: int) -> dict:
    report = {}
    for scope, rows in grouped.items():
        selected = rows["original"]
        prefix = "" if scope == "global" else f"{scope}_"
        count = 0 if scope == "with_stain" else zero_positive_images
        values = summarize_counts(flow_counts(rows, selected), int(selected.sum()), count)
        report.update((prefix + key, value) for key, value in values.items())
    return report


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_images(by_image: dict, baseline_by_id: dict, threshold: float | None) -> list[dict]:
    comparison = []
    for image_id, rows in by_image.items():
        original = rows["original"]
        selected = apply_threshold(rows["probability"], threshold)
        before, after = flow_counts(rows, original), flow_counts(rows, selected)
        if before != baseline_by_id[image_id]:
            raise ValueError(f"As decisões originais não reproduzem a baseline em {image_id}.")
        has_stain = bool(rows["stain"].any())
        before_report = summarize_counts(before, original.sum(), not has_stain and original.any())
        after_report = summarize_counts(after, selected.sum(), not has_stain and selected.any())
        comparison.append({"image_id": image_id, "has_stain": has_stain,
                           **{f"baseline_{key}": val for key, val in before_report.items()},
                           **{f"optimal_{key}": val for key, val in after_report.items()},
                           "iou_change": after_report["IoU"] - before_report["IoU"],
                           "fp_eliminated": int(rows["other"][original & ~selected].sum())})
    return comparison


def compare_i01(rows: np.ndarray, threshold: float | None) -> list[dict]:
    comparisons = []
    for region_id, note in I01_REGIONS.items():
        region, = rows[rows["id"] == region_id]
        after = bool(apply_threshold(np.array([region["probability"]]), threshold)[0])
        entry = {"image_id": I01, "region_id": region_id, "i01_label": note,
                 "probability": float(region["probability"]), "area": int(region["area"]),
                 "stain_occupancy": float(region["stain"] / region["area"]),
                 "baseline_prediction": int(region["original"]), "optimal_prediction": int(after)}
        for label, selected in (("baseline", bool(region["original"])), ("optimal", after)):
            counts = flow_counts(np.array([region]), np.array([selected]))
            entry.update((f"{label}_{key}", value) for key, value in counts.items())
        comparisons.append(entry)
    return comparisons


def save_charts(output: Path, grid: list, exact_curve: dict, baseline: dict, optimal: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    threshold = optimal["threshold"]
    grid_x = [row["threshold"] for row in grid]
    series = (
        ("performance_by_threshold", ("Precision", "Recall", "IoU", "F1"), "Métrica"),
        ("errors_by_threshold", ("FP", "FN"), "Pixels"),
        ("predicted_area_by_threshold", ("positive_pixels",), "Pixels previstos como Stain"),
        ("empty_images_by_threshold", ("without_stain_positive_images",), "Imagens sem Stain com previsão positiva"),
    )
    for filename, metrics, ylabel in series:
        figure, axis = plt.subplots(figsize=(9, 4))
        for metric in metrics:
            axis.plot(grid_x, [row[metric] for row in grid], label=metric)
        axis.axvline(0.50, color="gray", linestyle="--", label="0,50")
        if threshold is not None:
            axis.axvline(threshold, color="black", linestyle=":", label=f"Ótimo {threshold:.5f}")
        axis.set(xlabel="Limiar", ylabel=ylabel, xlim=(0, 1))
        axis.legend()
        save_chart(figure, output / f"{filename}.png")

    tp, fp, fn, tn = (exact_curve[key] for key in ("TP", "FP", "FN", "TN"))
    for filename, x, y, xkey, ykey, xlabel in (
        ("precision_recall", tp / np.maximum(tp + fn, 1), tp / np.maximum(tp + fp, 1),
         "Recall", "Precision", "Recall"),
        ("roc", fp / np.maximum(fp + tn, 1), tp / np.maximum(tp + fn, 1),
         "false_positive_rate", "Recall", "Taxa de falsos positivos"),
    ):
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(x, y, color="#356d91")
        for row, name, marker in ((baseline, "Baseline predict() · 0,50 nominal", "o"),
                                  (optimal, "Ótimo de IoU", "*")):
            axis.scatter(row[xkey], row[ykey], label=name, marker=marker, zorder=3)
        half = grid[50]
        axis.scatter(half[xkey], half[ykey], label="P ≥ 0,50", marker="x", zorder=4)
        axis.set(xlabel=xlabel, ylabel=ykey, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axis.legend()
        save_chart(figure, output / f"{filename}.png")

    metrics = ("Accuracy", "Precision", "Recall", "IoU", "F1", "Specificity")
    x = np.arange(len(metrics))
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(x - 0.2, [baseline[key] for key in metrics], width=0.4, label="Baseline predict()")
    axis.bar(x + 0.2, [optimal[key] for key in metrics], width=0.4, label="Ótimo de IoU")
    axis.set_xticks(x, metrics)
    axis.set_ylim(0, 1)
    axis.legend()
    save_chart(figure, output / "baseline_vs_optimal.png")


def save_chart(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def select_examples(comparison: list[dict]) -> dict[str, list[str]]:
    reasons = {I01: ["I01"]}
    improved = [row for row in comparison if row["has_stain"] and row["iou_change"] > 0]
    worsened = [row for row in comparison if row["has_stain"] and row["iou_change"] < 0]
    absent = [row for row in comparison if not row["has_stain"] and row["fp_eliminated"] > 0]
    for rows, key, description, sign in (
        (improved, "iou_change", "maior melhora de IoU", -1),
        (worsened, "iou_change", "maior piora de IoU", 1),
        (absent, "fp_eliminated", "sem Stain, mais FP eliminados", -1),
    ):
        for row in sorted(rows, key=lambda row: (sign * row[key], row["image_id"]))[:2]:
            reasons.setdefault(row["image_id"], []).append(description)
    return reasons


def save_examples(dataset: Path, output: Path, by_image: dict, comparison: list[dict],
                  slic: dict, threshold: float | None) -> dict:
    """Refaz apenas o SLIC dos exemplos para projetar as decisões salvas."""
    examples = select_examples(comparison)
    reports = {row["image_id"]: row for row in comparison}
    for image_id in examples:
        rows = np.sort(by_image[image_id], order="id")
        rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
        label = load_label(dataset / "Label" / f"{image_id}.png")
        regions = segment_regions(rgb, **slic)
        labeled = label_regions(regions, label)
        if (not np.array_equal(rows["id"], labeled.region_ids)
                or not np.array_equal(rows["area"], labeled.pixel_counts)
                or not np.array_equal(rows["stain"], labeled.target_counts)):
            raise ValueError(f"O SLIC do exemplo {image_id} divergiu do CSV do D02.")
        reference = create_binary_mask(label, DamageClass.STAIN)
        original = reconstruct_binary_mask(regions, rows["id"], rows["original"])
        optimal = reconstruct_binary_mask(regions, rows["id"], apply_threshold(rows["probability"], threshold))
        for name, mask in (("baseline", original), ("optimal", optimal)):
            if any(reports[image_id][f"{name}_{key}"] != count
                   for key, count in count_binary_pixels(reference, mask).items()):
                raise ValueError(f"A reconstrução visual divergiu em {image_id}, {name}.")
        figure, axes = comparison_panels([
            (rgb, "RGB"), (reference, "Ground Truth"), (original, "Baseline"),
            (optimal, "Ótimo global"), (pixel_error_map(reference, original), "Erros da baseline"),
            (pixel_error_map(reference, optimal), "Erros no ótimo"),
        ], columns=3)
        axes[-1].legend(handles=pixel_error_legend(), loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2)
        save_comparison(figure, output / f"{image_id}.png")
        plt.close(figure)
    return examples


def summarize_changes(regions: np.ndarray, comparison: list[dict], threshold: float | None) -> dict:
    original = regions["original"]
    selected = apply_threshold(regions["probability"], threshold)
    removed, added = original & ~selected, selected & ~original
    empty = regions["stain"] == 0
    with_stain = [row for row in comparison if row["has_stain"]]
    without_stain = [row for row in comparison if not row["has_stain"]]
    return {
        "false_positives_eliminated": int(regions["other"][removed].sum()),
        "false_positives_added": int(regions["other"][added].sum()),
        "new_false_negatives": int(regions["stain"][removed].sum()),
        "false_negatives_recovered": int(regions["stain"][added].sum()),
        "zero_occupancy_fp_baseline": int(regions["other"][original & empty].sum()),
        "zero_occupancy_fp_optimal": int(regions["other"][selected & empty].sum()),
        "with_stain_iou": {"improved": sum(row["iou_change"] > 0 for row in with_stain),
                          "worsened": sum(row["iou_change"] < 0 for row in with_stain),
                          "unchanged": sum(row["iou_change"] == 0 for row in with_stain)},
        "without_stain": {"images": len(without_stain),
                          "baseline_nonempty": sum(row["baseline_FP"] > 0 for row in without_stain),
                          "optimal_nonempty": sum(row["optimal_FP"] > 0 for row in without_stain)},
        "by_original_class": None,
        "by_original_class_reason": "O CSV do D02 só contém Stain e não Stain por superpixel.",
    }


def run_experiment(d02: Path, baseline: Path, dataset: Path, output: Path) -> dict:
    started = perf_counter()
    config = read_json(d02 / "configuration.json")
    ids = read_json(baseline / "split.json")["validation"]
    if len(ids) != 132 or len(set(ids)) != 132 or config["validation_ids"] != ids:
        raise ValueError("O D02 deve corresponder aos 132 IDs do manifesto de validação.")
    regions = read_regions(d02 / "region_statistics.csv")
    if (len(regions) != 103621 or set(regions["image"]) != set(ids)
            or len(np.unique(regions[["image", "id"]])) != len(regions)):
        raise ValueError("CSV do D02 com cobertura, contagem ou unicidade incompatível.")
    if flow_counts(regions, regions["original"]) != BASELINE_COUNTS:
        raise ValueError("O CSV do D02 não reproduz as contagens globais da baseline.")
    saved = read_json(baseline / "pixel_validation_per_image.json")
    baseline_by_id = {row["image_id"]: row["counts"] for row in saved}
    if len(saved) != len(ids) or set(baseline_by_id) != set(ids):
        raise ValueError("O relatório da baseline não corresponde à validação.")

    by_image = {image_id: regions[regions["image"] == image_id] for image_id in ids}
    empty_ids = [name for name, rows in by_image.items() if not rows["stain"].any()]
    is_empty = np.isin(regions["image"], empty_ids)
    grouped = {"global": regions, "with_stain": regions[~is_empty], "without_stain": regions[is_empty]}
    maxima = np.sort([by_image[name]["probability"].max() for name in empty_ids])
    original_empty_positive = sum(bool(by_image[name]["original"].any()) for name in empty_ids)
    baseline_report = baseline_groups(grouped, original_empty_positive)
    grid = np.arange(101) / 100
    exact = exact_thresholds(regions)
    curve = threshold_curve(regions, exact)
    affected = maxima.size - np.searchsorted(maxima, exact, side="left")
    optima = {
        key: ({"threshold": None if np.isinf(exact[i]) else float(exact[i]),
               "all_negative": bool(np.isinf(exact[i])), **curve_row(curve, i, affected[i])} if i is not None else None)
        for key, i in choose_optima(curve, BASELINE_COUNTS).items()
    }
    optimal = optima["max_iou"]
    threshold = optimal["threshold"]
    comparison = compare_images(by_image, baseline_by_id, threshold)
    for key in BASELINE_COUNTS:
        if (sum(row[f"optimal_{key}"] for row in comparison) != optimal[key]
                or sum(row[f"baseline_{key}"] for row in comparison) != BASELINE_COUNTS[key]):
            raise RuntimeError("As contagens individuais e globais divergem.")
    i01 = compare_i01(by_image[I01], threshold)
    half = apply_threshold(regions["probability"], 0.5)
    differences = half != regions["original"]
    changes = summarize_changes(regions, comparison, threshold)
    changes.update({"coverage_change_recall": optimal["Recall"] - baseline_report["Recall"],
                    "predicted_pixels_change": optimal["positive_pixels"] - baseline_report["positive_pixels"]})

    output.mkdir(parents=True, exist_ok=True)
    grid_rows = export_points(output / "threshold_metrics.csv", grouped, maxima, grid, exact, baseline_report)
    write_csv(output / "per_image_comparison.csv", comparison)
    write_csv(output / "i01_superpixel_comparison.csv", i01)
    write_json(output / "optimal_thresholds.json", optima)
    save_charts(output / "charts", grid_rows, curve, baseline_report, optimal)
    examples = save_examples(dataset, output / "examples", by_image, comparison,
                             config["pipeline_config"]["slic"], threshold)
    write_json(output / "configuration.json", {
        "validation_ids": ids, "regions_csv": (d02 / "region_statistics.csv").as_posix(),
        "slic_for_examples_only": config["pipeline_config"]["slic"],
        "grid_step": 0.01, "decision": "probability >= threshold", "tie_break": "highest threshold",
        "exact_distinct_probabilities": exact.size - 1,
        "baseline": "original_prediction from D02", "all_negative_threshold": None,
    })
    summary = {"images": len(ids), "regions": len(regions), "baseline": baseline_report,
               "grid": {str(value): grid_rows[int(round(value * 100))] for value in (0.5, 0.6, 0.7, 0.8, 0.9)},
               "literal_0_50_disagreement": {"regions": int(differences.sum()),
                   "pixels": int(regions["area"][differences].sum()),
                   "ties_at_0_50": int(np.count_nonzero(regions["probability"] == 0.5))},
               "changes": changes, "optima": optima, "i01": next(row for row in comparison if row["image_id"] == I01),
               "examples": examples, "wall_seconds": perf_counter() - started}
    write_json(output / "global_statistics.json", summary)
    write_summary(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d02", type=Path, default=Path("results/diagnostics/D02_false_positives"))
    parser.add_argument("--baseline", type=Path, default=Path("results/region_full"))
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/D03_threshold_analysis"))
    args = parser.parse_args()
    targets = tuple(path.name for path in args.output.iterdir()) if args.output.is_dir() else ()
    guard_cli_output(args.output, targets, owner="D03_threshold_analysis")
    report = run_experiment(args.d02, args.baseline, args.dataset, args.output)
    print(json.dumps({key: report[key] for key in ("optima", "changes", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
