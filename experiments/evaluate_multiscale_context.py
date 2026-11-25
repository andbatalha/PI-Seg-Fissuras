"""E01.1: contexto Lab multiescala em regiões da baseline de Stain."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

from experiments.evaluate_thresholds import (
    BASELINE_COUNTS, I01, I01_REGIONS, REGION_DTYPE, apply_threshold, flow_counts,
    optimal_threshold, read_json, read_regions, summarize_counts, write_csv, write_json,
)
from experiments.reporting import guard_cli_output, write_summary
from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.dataset.masks import create_binary_mask
from src.evaluation import count_binary_pixels, count_region_pixels, pixel_metrics
from src.feature_extractor import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS
from src.multiscale_context import MULTISCALE_NAMES
from src.region_dataset import build_image_rows, save_region_dataset
from src.region_features import FEATURE_NAMES
from src.region_prediction import forest_decisions, reconstruct_binary_mask
from src.region_training import FOREST_CONFIG, binary_samples, fit_region_forest
from src.segmentation import segment_regions
from visualization.plotting import comparison_panels, pixel_error_legend, pixel_error_map, save_comparison


VARIANTS = {"B00": 15, "E01.1-A": 21, "E01.1-B": 25}
VARIANT_GROUPS = {
    "B00": DEFAULT_FEATURE_GROUPS,
    "E01.1-A": (*DEFAULT_FEATURE_GROUPS, "multiscale_lab"),
    "E01.1-B": tuple(FEATURE_GROUPS),
}
FIXED_THRESHOLD = 0.7404814731187024
POLICIES = ("default", "fixed_d03", "optimal_iou")
HOMOGENEOUS_FEATURES = ("glcm_contrast", "hist_entropy", "mean_gradient", "std_l")


def check_alignment(saved: dict, indices: np.ndarray, rows: dict) -> None:
    """Rejeita reutilização quando IDs, áreas, contagens ou rótulos divergem."""
    for key in ("region_ids", "pixel_counts", "target_counts", "target_fractions", "labels"):
        if not np.array_equal(saved[key][indices], rows[key]):
            raise ValueError(f"O campo {key} diverge do dataset original.")


def prepare_split(dataset: Path, original_path: Path, output: Path, ids: list[str], config: dict,
                  *, validation: bool) -> dict:
    """Extrai contexto uma vez e mantém metadados, ordem e amostras da baseline."""
    started = perf_counter()
    with np.load(original_path, allow_pickle=False) as source:
        saved = dict(source.items())
    if (saved["image_ids"].tolist() != ids or tuple(saved["feature_names"]) != FEATURE_NAMES
            or json.loads(saved["pipeline_config"].item()) != config):
        raise ValueError(f"Metadados incompatíveis em {original_path}.")
    columns = np.empty((len(saved["labels"]), len(MULTISCALE_NAMES)))
    times = np.zeros((len(ids), 3))
    class_counts = np.empty((len(columns), len(DamageClass)), dtype=np.int64) if validation else None
    groups = tuple(FEATURE_GROUPS) if validation else ("multiscale_lab", "multiscale_robust")
    for i, image_id in enumerate(ids):
        rows = np.flatnonzero(saved["image_indices"] == i)
        timing = {}
        extracted = build_image_rows(dataset, image_id, feature_groups=groups, pipeline_config=config,
                                     include_class_counts=validation, timings=timing)
        check_alignment(saved, rows, extracted)
        context = extracted["features"]
        if validation:
            original, context = np.split(context, [len(FEATURE_NAMES)], axis=1)
            if not np.array_equal(original, saved["features"][rows]):
                raise ValueError(f"As 15 características originais divergem em {image_id}.")
            class_counts[rows] = extracted["class_pixel_counts"]
        columns[rows] = context
        times[i] = timing["rgb_slic"] + timing["base"], timing["multiscale"], timing["robust"]
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(ids):
            print(f"{original_path.stem}: {i + 1}/{len(ids)}", flush=True)

    features = np.column_stack((saved.pop("features"), columns))
    if validation:
        saved["class_pixel_counts"] = class_counts
    saved.update(features=features, inference_preparation_seconds=times)
    save_region_dataset(output, saved, FEATURE_GROUPS)
    return {"images": len(ids), "regions": len(features), "seconds": perf_counter() - started,
            "bytes": output.stat().st_size}


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive.items())


def model_metadata(baseline: Path) -> tuple[dict, dict]:
    manifest = read_json(baseline / "split.json")
    artifact = joblib.load(baseline / "random_forest.joblib")
    if (len(manifest["train"]) != 539 or len(manifest["validation"]) != 132
            or len(set(manifest["train"] + manifest["validation"])) != 671
            or artifact["image_ids"] != manifest["train"] or tuple(artifact["feature_names"]) != FEATURE_NAMES
            or artifact["forest_config"] != FOREST_CONFIG
            or any(artifact["model"].get_params()[key] != value for key, value in FOREST_CONFIG.items())):
        raise ValueError("Manifesto, características ou hiperparâmetros incompatíveis com a baseline.")
    return manifest, artifact


def prepare_datasets(dataset: Path, baseline: Path, output: Path) -> None:
    manifest, artifact = model_metadata(baseline)
    root = output / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    statistics = {}
    for name in ("train", "validation"):
        statistics[name] = prepare_split(dataset, baseline / f"{name}.npz", root / f"{name}.npz",
                                         manifest[name], artifact["pipeline_config"], validation=name == "validation")
    write_json(root / "preparation_statistics.json", statistics)
    write_summary(root)


def validate_prepared(saved: dict, original: dict) -> None:
    """Garante que só houve adição de colunas, sem trocar linhas ou metadados."""
    if tuple(saved["feature_names"]) != (*FEATURE_NAMES, *MULTISCALE_NAMES):
        raise ValueError("Ordem das características experimentais incompatível.")
    if "feature_groups" in saved and saved["feature_groups"].tolist() != list(FEATURE_GROUPS):
        raise ValueError("Grupos de características experimentais incompatíveis.")
    for key in ("image_ids", "image_indices", "region_ids", "pixel_counts", "target_counts",
                "labels", "pipeline_config"):
        if not np.array_equal(saved[key], original[key]):
            raise ValueError(f"O dataset experimental mudou o campo {key}.")
    if not np.array_equal(saved["features"][:, :len(FEATURE_NAMES)], original["features"]):
        raise ValueError("As características originais foram modificadas.")


def train_variants(data: dict, artifact: dict, output: Path,
                   variants: tuple = tuple(VARIANTS), *, reuse: bool = False) -> tuple[dict, dict]:
    """Ajusta as variantes independentes com as mesmas linhas binárias e parâmetros."""
    models = {"B00": artifact["model"]}
    features, labels, dataset_metadata = binary_samples(data, source="treino do E01.1")
    if dataset_metadata["feature_groups"] != list(FEATURE_GROUPS):
        raise ValueError("Grupos de características incompatíveis com o dataset compartilhado do E01.1.")
    statistics = {}
    parameters = artifact["model"].get_params()
    if not reuse:
        output.mkdir(parents=True, exist_ok=True)
    for name in variants:
        columns = VARIANTS[name]
        metadata = {"variant": name, "forest_config": parameters, "pipeline_config": artifact["pipeline_config"],
                    "feature_names": dataset_metadata["feature_names"][:columns],
                    "image_ids": dataset_metadata["image_ids"]}
        groups = list(VARIANT_GROUPS[name])
        row = {"training_examples": labels.size, "negative": int((labels == 0).sum()),
               "positive": int((labels == 1).sum()), "features": columns, "forest_config": parameters}
        if name != "B00":
            path = output / f"{name}.joblib"
            if reuse:
                saved = joblib.load(path)
                model = saved["model"]
                if (any(saved.get(key) != value for key, value in metadata.items())
                        or saved.get("feature_groups", groups) != groups
                        or model.n_features_in_ != columns or model.get_params() != parameters):
                    raise ValueError(f"Modelo incompatível: {path}")
                row["training_seconds"] = saved.get("training_seconds")
            else:
                model, row["training_seconds"] = fit_region_forest(
                    features[:, :columns], labels, forest_config=parameters,
                )
                joblib.dump({"model": model, **metadata, "feature_groups": groups,
                             "training_seconds": row["training_seconds"]}, path, compress=3)
                print(f"Treino concluído: {name} ({row['training_seconds']:.2f} s)", flush=True)
            row["model_bytes"] = path.stat().st_size
            row["model_path"] = path.as_posix()
            models[name] = model
        statistics[name] = row
    return models, statistics


def homogeneous_definition(d04: Path) -> dict:
    """Congela a regra de homogeneidade nos primeiros quartis dos TP da baseline."""
    with (d04 / "feature_statistics_by_outcome.csv").open(encoding="utf-8", newline="") as stream:
        bounds = {row["feature"]: float(row["q1"]) for row in csv.DictReader(stream)
                  if row["group"] == "TP" and row["feature"] in HOMOGENEOUS_FEATURES}
    if set(bounds) != set(HOMOGENEOUS_FEATURES):
        raise ValueError("Os quartis dos TP necessários não estão registrados no D04.")
    return bounds


def evaluation_rows(data: dict) -> np.ndarray:
    rows = np.empty(len(data["labels"]), dtype=REGION_DTYPE)
    rows["image"] = data["image_ids"][data["image_indices"]]
    rows["id"] = data["region_ids"]
    rows["area"] = data["pixel_counts"]
    rows["stain"] = data["target_counts"]
    rows["other"] = rows["area"] - rows["stain"]
    return rows


def score_model(model, data: dict, columns: int, locations: list[np.ndarray],
                *, progress: str = "") -> tuple[np.ndarray, ...]:
    """Prediz por imagem e reaproveita as probabilidades em todas as políticas."""
    features = data["features"]
    if columns != model.n_features_in_ or features.shape[1] < columns:
        raise ValueError("O NPZ não fornece a quantidade de colunas esperada pelo estimador.")
    probabilities = np.empty(len(features))
    selected = np.empty(len(features), dtype=bool)
    times = np.empty(len(locations))
    for i, rows in enumerate(locations):
        started = perf_counter()
        predicted, scores = forest_decisions(model, model.predict_proba(features[rows, :columns]))
        times[i] = perf_counter() - started
        probabilities[rows], selected[rows] = scores, predicted
        if progress and ((i + 1) % 25 == 0 or i + 1 == len(locations)):
            print(f"{progress} · {i + 1}/{len(locations)} imagens · "
                  f"{sum(map(len, locations[:i + 1]))} regiões", flush=True)
    return selected, probabilities, times


def evaluate_decision(rows: np.ndarray, selected: np.ndarray, locations: list[np.ndarray], times: np.ndarray,
                      model_name: str, policy: str, threshold: float | None) -> tuple[dict, list[dict]]:
    per_image = []
    for indices, seconds in zip(locations, times):
        group = rows[indices]
        flags = selected[indices]
        counts = count_region_pixels(group["stain"], group["other"], flags)
        has_stain = bool(group["stain"].any())
        per_image.append({"variant": model_name, "policy": policy, "threshold": threshold,
                          "image_id": str(group["image"][0]), "has_stain": has_stain,
                          **summarize_counts(counts, flags.sum(), not has_stain and flags.any()),
                          "inference_seconds": float(seconds)})
    counts = {key: sum(row[key] for row in per_image) for key in BASELINE_COUNTS}
    if counts != count_region_pixels(rows["stain"], rows["other"], selected):
        raise ValueError("As contagens por imagem divergem das contagens globais.")
    with_stain = [row["IoU"] for row in per_image if row["has_stain"]]
    report = {"variant": model_name, "policy": policy, "threshold": threshold,
              **summarize_counts(counts, selected.sum(), sum(row["without_stain_positive_images"] for row in per_image)),
              "mean_iou_with_stain": float(np.mean(with_stain)), "median_iou_with_stain": float(np.median(with_stain)),
              "mean_inference_seconds": float(times.mean())}
    return report, per_image


def false_positive_rows(selected: np.ndarray, class_counts: np.ndarray, variant: str, policy: str) -> list[dict]:
    totals, predicted = class_counts.sum(axis=0), class_counts[selected].sum(axis=0)
    false_positives = int(predicted[:int(DamageClass.STAIN)].sum())
    return [{"variant": variant, "policy": policy, "class_id": int(damage), "class": damage.name,
             "pixels": int(totals[damage]), "FP": int(predicted[damage]),
             "fp_rate": float(predicted[damage] / totals[damage]) if totals[damage] else 0.0,
             "fp_share": float(predicted[damage] / false_positives) if false_positives else 0.0}
            for damage in DamageClass if damage != DamageClass.STAIN]


def specific_errors(rows: np.ndarray, selected: np.ndarray, baseline: np.ndarray,
                    class_counts: np.ndarray, predominant: np.ndarray, homogeneous: np.ndarray) -> dict:
    erosion = 10 * class_counts[:, DamageClass.EROSION] >= 7 * rows["area"]
    return {"zero_occupancy_fp": int(rows["other"][selected & (rows["stain"] == 0)].sum()),
            "erosion_fp": int(class_counts[selected, DamageClass.EROSION].sum()),
            "predominant_erosion_regions": int(erosion.sum()),
            "predominant_erosion_predicted_positive": int((erosion & selected).sum()),
            "predominant_stain_regions": int(predominant.sum()),
            "predominant_stain_recovered": int((predominant & selected & ~baseline).sum()),
            "predominant_stain_lost": int((predominant & ~selected & baseline).sum()),
            "homogeneous_stain_regions": int(homogeneous.sum()),
            "homogeneous_stain_detected": int((homogeneous & selected).sum()),
            "homogeneous_stain_recovered": int((homogeneous & selected & ~baseline).sum()),
            "homogeneous_stain_lost": int((homogeneous & ~selected & baseline).sum())}


def check_baseline(rows: np.ndarray, d02: Path) -> None:
    saved = read_regions(d02 / "region_statistics.csv")
    aligned = np.sort(saved, order=("image", "id"))
    order = np.argsort(rows, order=("image", "id"))
    for key in ("image", "id", "area", "stain", "other", "original"):
        if not np.array_equal(rows[key][order], aligned[key]):
            raise ValueError(f"O campo {key} da baseline diverge do D02.")
    if not np.allclose(rows["probability"][order], aligned["probability"], rtol=0, atol=1e-12):
        raise ValueError("As probabilidades refeitas da baseline divergem do D02.")
    # A decisão no corte exato usa os valores históricos que definiram aquele ponto.
    rows["probability"][order] = aligned["probability"]


def export_thresholds(path: Path, curves: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for name, (thresholds, curve) in curves.items():
            for i, threshold in enumerate(thresholds):
                counts = {key: int(curve[key][i]) for key in BASELINE_COUNTS}
                row = {"variant": name, "threshold": None if np.isinf(threshold) else float(threshold),
                       "all_negative": bool(np.isinf(threshold)), **pixel_metrics(counts), **counts,
                       "positive_regions": int(curve["positive_regions"][i]),
                       "positive_pixels": counts["TP"] + counts["FP"]}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)


def write_i01(output: Path, rows_by_model: dict, selections: dict) -> None:
    items = []
    for name, rows in rows_by_model.items():
        for region_id, original_outcome in I01_REGIONS.items():
            index, = np.flatnonzero((rows["image"] == I01) & (rows["id"] == region_id))
            row = rows[index]
            items.append({"variant": name, "image_id": I01, "region_id": region_id,
                          "i01_outcome": original_outcome, "area": int(row["area"]),
                          "stain_occupancy": float(row["stain"] / row["area"]),
                          "probability": float(row["probability"]),
                          **{policy: int(selections[name][policy][index]) for policy in POLICIES}})
    write_csv(output, items)


def examples_to_save(d03: dict, per_image: list[dict]) -> dict:
    examples = {image_id: ["referência fixa D03"] for image_id in d03["examples"]}
    examples.setdefault(I01, []).append("I01")
    baseline = {row["image_id"]: row["IoU"] for row in per_image
                if row["variant"] == "B00" and row["policy"] == "default"}
    for name in ("E01.1-A", "E01.1-B"):
        gains = [(row["IoU"] - baseline[row["image_id"]], row["image_id"]) for row in per_image
                 if row["variant"] == name and row["policy"] == "default" and row["has_stain"]]
        improving = sorted((pair for pair in gains if pair[0] > 0), key=lambda p: (-p[0], p[1]))
        declining = sorted(pair for pair in gains if pair[0] < 0)
        for items, reason in ((improving, "maior ganho"), (declining, "maior perda")):
            if items:
                examples.setdefault(items[0][1], []).append(f"{name}: {reason} de IoU · decisão padrão")
    return examples


def save_examples(dataset: Path, output: Path, examples: dict, locations: dict, rows: np.ndarray,
                  selections: dict, per_image: list[dict], config: dict) -> None:
    expected = {(row["variant"], row["policy"], row["image_id"]): row for row in per_image}
    for image_id in examples:
        rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
        regions = segment_regions(rgb, **config["slic"])
        label = load_label(dataset / "Label" / f"{image_id}.png")
        reference = create_binary_mask(label, DamageClass.STAIN)
        subset = locations[image_id]
        for policy in ("default", "optimal_iou"):
            masks, errors = [], []
            for name, variant in selections.items():
                mask = reconstruct_binary_mask(regions, rows["id"][subset], variant[policy][subset])
                if any(expected[name, policy, image_id][key] != count
                       for key, count in count_binary_pixels(reference, mask).items()):
                    raise ValueError(f"A reconstrução do exemplo {image_id} divergiu em {name}, {policy}.")
                masks.append((mask, name))
                errors.append((pixel_error_map(reference, mask), f"Erros · {name}"))
            figure, axes = comparison_panels([
                (rgb, "RGB"), (reference, "Ground Truth"), (mark_boundaries(rgb, regions), "SLIC"),
                *masks, *errors,
            ], columns=3)
            axes[-1].legend(handles=pixel_error_legend(), loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2)
            save_comparison(figure, output / f"{image_id}_{policy}.png")
            plt.close(figure)


def save_charts(output: Path, comparison: list[dict], curves: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(curves))
    for i, policy in enumerate(POLICIES):
        values = [row["IoU"] for row in comparison if row["policy"] == policy]
        axis.bar(x + (i - 1) * 0.22, values, width=0.22, label=policy)
    axis.set(xticks=x, xticklabels=list(curves), ylabel="IoU global")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "iou_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    for name, (thresholds, curve) in curves.items():
        union = curve["TP"] + curve["FP"] + curve["FN"]
        axis.plot(thresholds[1:], (curve["TP"] / np.maximum(union, 1))[1:], label=name)
    axis.axvline(FIXED_THRESHOLD, color="gray", linestyle="--", label="D03")
    axis.set(xlabel="Limiar", ylabel="IoU global", xlim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "iou_by_threshold.png", dpi=150)
    plt.close(figure)


def train_prepared(baseline: Path, prepared: Path, output: Path, variants: tuple) -> dict:
    """Treina só as variantes solicitadas a partir do NPZ já preparado."""
    _, artifact = model_metadata(baseline)
    train = read_npz(prepared / "train.npz")
    validate_prepared(train, read_npz(baseline / "train.npz"))
    _, statistics = train_variants(train, artifact, output / "models", variants)
    report_path = output / "training_statistics.json"
    existing = read_json(report_path) if report_path.is_file() else {}
    write_json(report_path, {**existing, **statistics})
    write_summary(output)
    return statistics


def run_evaluation(dataset: Path, baseline: Path, diagnostics: Path, output: Path, *,
                   variants: tuple = tuple(VARIANTS), prepared: Path | None = None,
                   models_root: Path | None = None, reuse_models: bool = False,
                   render: bool = True) -> dict:
    started = perf_counter()
    manifest, artifact = model_metadata(baseline)
    config = artifact["pipeline_config"]
    baseline_train = read_json(baseline / "validation_metrics.json")
    d03 = read_json(diagnostics / "D03_threshold_analysis" / "global_statistics.json")
    bounds = homogeneous_definition(diagnostics / "D04_feature_importance")
    variants = ("B00", *(name for name in variants if name != "B00"))
    prepared = prepared or output / "datasets"
    train, validation = (read_npz(prepared / f"{name}.npz") for name in ("train", "validation"))
    for name, split in (("train", train), ("validation", validation)):
        validate_prepared(split, read_npz(baseline / f"{name}.npz"))
    if int((train["labels"] != -1).sum()) != baseline_train["training_regions"]:
        raise ValueError("Quantidade de amostras binárias de treino divergente da baseline.")
    models, training = train_variants(train, artifact, models_root or output / "models", variants, reuse=reuse_models)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "configuration.json", {
        "train_ids": manifest["train"], "validation_ids": manifest["validation"], "pipeline_config": config,
        "forest_config": artifact["model"].get_params(), "fixed_threshold": FIXED_THRESHOLD,
        "features": {name: validation["feature_names"][:VARIANTS[name]].tolist() for name in variants},
        "homogeneity_rule": "ocupação Stain >= 70% e todas as quatro features <= Q1 de TP do D04",
        "homogeneity_bounds": bounds, "tie_break": "maior limiar, incluindo solução totalmente negativa",
    })
    training["B00"].update(training_seconds=baseline_train["training_seconds"],
                           model_bytes=(baseline / "random_forest.joblib").stat().st_size,
                           model_path=(baseline / "random_forest.joblib").as_posix())
    write_json(output / "training_statistics.json", training)

    locations = {image_id: np.flatnonzero(validation["image_indices"] == i)
                 for i, image_id in enumerate(manifest["validation"])}
    rows = evaluation_rows(validation)
    predominant = 10 * rows["stain"] >= 7 * rows["area"]
    homogeneous = predominant.copy()
    for feature, bound in bounds.items():
        homogeneous &= validation["features"][:, FEATURE_NAMES.index(feature)] <= bound
    class_counts = validation["class_pixel_counts"]
    if not np.array_equal(class_counts.sum(axis=1), rows["area"]) or not np.array_equal(class_counts[:, 5], rows["stain"]):
        raise ValueError("Contagens multiclasse incompatíveis com áreas e anotação de Stain.")
    if len(rows) != 103621 or sum(rows["area"]) != 43253760 or sum(
            bool(rows["stain"][indices].any()) for indices in locations.values()) != 75:
        raise ValueError("As dimensões da validação divergem da baseline.")

    comparison, per_image, fp_classes = [], [], []
    selections, rows_by_model, curves = {}, {}, {}
    base, context, robust = validation["inference_preparation_seconds"].T
    preparation = {"B00": base, "E01.1-A": base + context - robust, "E01.1-B": base + context}
    for name in variants:
        columns = VARIANTS[name]
        result = rows.copy()
        result["original"], result["probability"], inference = score_model(
            models[name], validation, columns, list(locations.values()), progress=name,
        )
        if name == "B00":
            check_baseline(result, diagnostics / "D02_false_positives")
        optimal, thresholds, curve = optimal_threshold(result)
        if name == "B00" and (optimal != FIXED_THRESHOLD or flow_counts(result, result["original"]) != BASELINE_COUNTS):
            raise ValueError("A baseline não reproduziu os totais originais ou o limiar do D03.")
        rows_by_model[name], curves[name] = result, (thresholds, curve)
        policies = {"default": None, "fixed_d03": FIXED_THRESHOLD, "optimal_iou": optimal}
        selections[name] = {policy: result["original"] if policy == "default" else apply_threshold(result["probability"], cut)
                            for policy, cut in policies.items()}
        for policy, selected in selections[name].items():
            report, records = evaluate_decision(result, selected, list(locations.values()),
                                               preparation[name] + inference, name, policy, policies[policy])
            if name == "B00" and policy != "default" and any(
                    report[key] != d03["optima"]["max_iou"][key] for key in BASELINE_COUNTS):
                raise ValueError("As contagens da baseline no limiar fixo não reproduziram o D03.")
            report["mean_classifier_seconds"] = float(inference.mean())
            report.update(specific_errors(result, selected, selections["B00"][policy], class_counts, predominant, homogeneous))
            comparator = report if name == "B00" else next(row for row in comparison if row["variant"] == "B00" and row["policy"] == policy)
            report["iou_delta_vs_b00"] = report["IoU"] - comparator["IoU"]
            comparison.append(report)
            per_image.extend(records)
            class_results = false_positive_rows(selected, class_counts, name, policy)
            if sum(row["FP"] for row in class_results) != report["FP"]:
                raise ValueError("As contagens de falsos positivos por classe divergem do total.")
            fp_classes.extend(class_results)
        print(f"Validação concluída: {name}", flush=True)

    examples = examples_to_save(d03, per_image)
    write_csv(output / "model_comparison.csv", comparison)
    write_csv(output / "per_image_metrics.csv", per_image)
    write_csv(output / "false_positives_by_class.csv", fp_classes)
    write_i01(output / "i01_superpixel_comparison.csv", rows_by_model, selections)
    export_thresholds(output / "threshold_metrics.csv", curves)
    np.savez_compressed(output / "predictions.npz", **{f"rows_{name}": result for name, result in rows_by_model.items()},
                        **{f"selection_{name}_{policy}": values for name, policies in selections.items()
                           for policy, values in policies.items()})
    if render:
        save_charts(output / "charts", comparison, curves)
        save_examples(dataset, output / "examples", examples, locations, rows, selections, per_image, config)
    report = {"images": len(locations), "regions": len(rows), "examples": examples, "comparison": comparison,
              "dataset_preparation": read_json(prepared / "preparation_statistics.json"),
              "train_evaluate_export_seconds": perf_counter() - started}
    write_json(output / "global_metrics.json", report)
    write_summary(output)
    return report


def render_saved_evaluation(dataset: Path, output: Path) -> None:
    """Regenera figuras a partir das decisões salvas, sem carregar o Random Forest."""
    config = read_json(output / "configuration.json")
    report = read_json(output / "global_metrics.json")
    predictions = read_npz(output / "predictions.npz")
    rows = predictions["rows_B00"]
    locations = {name: np.flatnonzero(rows["image"] == name) for name in config["validation_ids"]}
    variants = tuple(config["features"])
    selections = {name: {policy: predictions[f"selection_{name}_{policy}"] for policy in POLICIES}
                  for name in variants}
    if (set(rows["image"]) != set(locations) or not set(report["examples"]).issubset(locations)
            or any(values.shape != rows.shape for policies in selections.values() for values in policies.values())):
        raise ValueError("IDs ou dimensões incompatíveis no cache de predições.")
    with (output / "per_image_metrics.csv").open(encoding="utf-8", newline="") as stream:
        per_image = [{**row, **{key: int(row[key]) for key in BASELINE_COUNTS}} for row in csv.DictReader(stream)]
    curves = {name: optimal_threshold(predictions[f"rows_{name}"])[1:] for name in variants}
    save_charts(output / "charts", report["comparison"], curves)
    save_examples(dataset, output / "examples", report["examples"], locations, rows,
                  selections, per_image, config["pipeline_config"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--baseline", type=Path, default=Path("results/region_full"))
    parser.add_argument("--diagnostics", type=Path, default=Path("results/diagnostics"))
    parser.add_argument("--output", type=Path, default=Path("results/optimization/E01_1_multiscale_context"))
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate", "train", "validate", "render"), default="all")
    parser.add_argument("--variant", choices=("all", "A", "B"), default="all")
    parser.add_argument("--prepared", type=Path, help="Diretório de NPZ experimentais já preparados.")
    parser.add_argument("--models", type=Path, help="Diretório dos modelos para a etapa validate.")
    args = parser.parse_args()
    variants = tuple(VARIANTS) if args.variant == "all" else (f"E01.1-{args.variant}",)
    prepared = args.prepared or args.output / "datasets"
    evaluation_targets = ("configuration.json", "training_statistics.json", "global_metrics.json", "predictions.npz",
                          "model_comparison.csv", "per_image_metrics.csv", "false_positives_by_class.csv",
                          "i01_superpixel_comparison.csv", "threshold_metrics.csv", "summary.md")
    model_targets = tuple(f"models/{name}.joblib" for name in variants if name != "B00")
    targets = set()
    if args.phase in ("all", "prepare"):
        targets.add("datasets")
    if args.phase in ("all", "evaluate", "train"):
        targets.update((*model_targets, "training_statistics.json", "summary.md"))
    if args.phase in ("all", "evaluate", "validate"):
        targets.update(evaluation_targets)
    if args.phase in ("all", "evaluate", "render"):
        targets.update(("charts", "examples", "summary.md"))
    guard_cli_output(args.output, tuple(sorted(targets)), owner="E01_1_multiscale_context")
    if args.phase in ("all", "prepare"):
        prepare_datasets(args.dataset, args.baseline, args.output)
    if args.phase == "train":
        train_prepared(args.baseline, prepared, args.output, tuple(name for name in variants if name != "B00"))
    if args.phase in ("all", "evaluate", "validate"):
        report = run_evaluation(args.dataset, args.baseline, args.diagnostics, args.output,
                                variants=variants, prepared=prepared,
                                models_root=args.models if args.phase == "validate" else None,
                                reuse_models=args.phase == "validate", render=args.phase != "validate")
        print(json.dumps(report, indent=2))
    if args.phase == "render":
        render_saved_evaluation(args.dataset, args.output)


if __name__ == "__main__":
    main()
