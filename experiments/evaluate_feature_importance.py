"""Diagnóstico da dependência do Random Forest das características de Stain."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, rankdata

from experiments.evaluate_thresholds import BASELINE_COUNTS, I01, I01_REGIONS, read_json, read_regions, write_csv, write_json
from experiments.reporting import guard_cli_output, write_summary
from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.evaluation import count_region_pixels, pixel_metrics
from src.region_features import FEATURE_GROUPS, FEATURE_NAMES, _LEVELS, extract_region_features
from src.region_labeling import label_regions
from src.region_prediction import forest_decisions
from src.segmentation import segment_regions


GROUPS = dict(zip(("Cor CIELab", "Contexto espacial", "Histograma", "Textura GLCM", "Gradiente"),
                  FEATURE_GROUPS.values()))
FEATURE_INDEX = {name: position for position, name in enumerate(FEATURE_NAMES)}
D03_THRESHOLD = 0.7404814731187024
REPEATS = 5
SEED = 42
PREDOMINANCE = 0.70
PROBABILITY_TOLERANCE = 1e-12
METRICS = ("IoU", "F1", "Precision", "Recall", "FP", "FN")
H1 = {"glcm_contrast": -1, "glcm_homogeneity": 1, "glcm_asm": 1,
      "hist_entropy": -1, "mean_gradient": -1, "std_l": -1}


def align_regions(expected: np.ndarray, ids: np.ndarray, areas: np.ndarray, stain: np.ndarray) -> np.ndarray:
    """Reordena o D02 pelos IDs da extração e valida as contagens de cada região."""
    sorted_rows = np.sort(expected, order="id")
    if (not np.array_equal(sorted_rows["id"], ids) or not np.array_equal(sorted_rows["area"], areas)
            or not np.array_equal(sorted_rows["stain"], stain)):
        raise ValueError("Os IDs, áreas ou contagens de Stain divergem do D02.")
    return sorted_rows


def extract_validation(dataset: Path, output: Path, ids: list, reference: np.ndarray, config: dict) -> dict:
    """Extrai os vetores uma vez e associa as anotações somente após a extração."""
    matrices, rows, labels, compositions = [], [], [], []
    for position, image_id in enumerate(ids, 1):
        rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
        regions = segment_regions(rgb, **config["slic"])
        region_ids, features = extract_region_features(
            rgb, regions, neighborhood_radius=config["features"]["neighborhood_radius"],
        )
        annotation = load_label(dataset / "Label" / f"{image_id}.png")
        labeling = label_regions(regions, annotation, **config["labeling"])
        if not np.array_equal(region_ids, labeling.region_ids):
            raise ValueError(f"IDs extraídos e anotados divergem em {image_id}.")
        expected = reference[reference["image"] == image_id]
        rows.append(align_regions(expected, region_ids, labeling.pixel_counts, labeling.target_counts))
        if not np.isin(annotation, [int(damage) for damage in DamageClass]).all():
            raise ValueError(f"Classe BFDD inválida em {image_id}.")
        bins = labeling.pixel_indices.ravel() * len(DamageClass) + annotation.ravel()
        compositions.append(np.bincount(bins, minlength=len(region_ids) * len(DamageClass)).reshape(-1, len(DamageClass)))
        matrices.append(features)
        labels.append(labeling.labels)
        if position == 1 or position % 20 == 0 or position == len(ids):
            print(f"Extração D04: {position}/{len(ids)}", flush=True)
    regions = np.concatenate(rows)
    data = {"features": np.concatenate(matrices), "regions": regions,
            "training_labels": np.concatenate(labels), "class_pixel_counts": np.concatenate(compositions),
            "stain_occupancy": regions["stain"] / regions["area"], "feature_names": np.array(FEATURE_NAMES)}
    np.savez_compressed(output / "validation_features.npz", **data)
    return data


def permute_columns(features: np.ndarray, columns: tuple[int, ...], seed: int) -> np.ndarray:
    """Embaralha a mesma linha para todas as colunas do grupo e preserva a entrada."""
    order = np.random.default_rng(seed).permutation(len(features))
    permuted = features.copy()
    permuted[:, columns] = features[np.ix_(order, columns)]
    return permuted


def weighted_metrics(regions: np.ndarray, selected: np.ndarray) -> dict:
    """Avalia por pixel sem reconstruir máscaras, usando as contagens por região."""
    counts = count_region_pixels(regions["stain"], regions["other"], selected)
    return {**counts, **pixel_metrics(counts)}


def verify_reference(model, features: np.ndarray, regions: np.ndarray, baseline: Path, d03: Path) -> tuple[dict, dict]:
    default, probabilities = forest_decisions(model, model.predict_proba(features))
    if not np.array_equal(default, model.predict(features)) or not np.array_equal(default, regions["original"]):
        raise ValueError("As decisões recalculadas diferem da baseline/D02.")
    drift = np.abs(probabilities - regions["probability"])
    if drift.max() > PROBABILITY_TOLERANCE:
        raise ValueError("As probabilidades extraídas não correspondem ao D02.")
    stored_optimal = regions["probability"] >= D03_THRESHOLD
    computed_optimal = probabilities >= D03_THRESHOLD
    changed = computed_optimal != stored_optimal
    if (np.abs(regions["probability"][changed] - D03_THRESHOLD) > PROBABILITY_TOLERANCE).any():
        raise ValueError("O desacordo no corte D03 excede a tolerância de ponto flutuante.")

    # A referência do D03 usa os valores históricos do D02, inclusive no corte exato.
    references = {"default": weighted_metrics(regions, default), "d03": weighted_metrics(regions, stored_optimal)}
    d03_report = read_json(d03 / "optimal_thresholds.json")["max_iou"]
    if d03_report["threshold"] != D03_THRESHOLD or any(references["d03"][key] != d03_report[key] for key in BASELINE_COUNTS):
        raise ValueError("Os totais de referência não reproduzem o D03.")
    if any(references["default"][key] != expected for key, expected in BASELINE_COUNTS.items()):
        raise ValueError("Os totais de referência não reproduzem a baseline.")

    baseline_rows = {row["image_id"]: row["counts"] for row in read_json(baseline / "pixel_validation_per_image.json")}
    with (d03 / "per_image_comparison.csv").open(encoding="utf-8", newline="") as stream:
        d03_rows = {row["image_id"]: row for row in csv.DictReader(stream)}
    for image_id in np.unique(regions["image"]):
        mask = regions["image"] == image_id
        for label, selected in (("default", default), ("d03", stored_optimal)):
            scores = weighted_metrics(regions[mask], selected[mask])
            expected = baseline_rows[image_id] if label == "default" else {
                key: int(d03_rows[image_id][f"optimal_{key}"]) for key in BASELINE_COUNTS
            }
            if any(scores[key] != expected[key] for key in BASELINE_COUNTS):
                raise ValueError(f"Contagens individuais divergentes em {image_id}, {label}.")
    return references, {"max_probability_drift": float(drift.max()), "d03_float_boundary_rows": int(changed.sum())}


def native_importance(model) -> tuple[list[dict], list[dict]]:
    native = np.asarray(model.feature_importances_)
    forest = np.stack([tree.feature_importances_ for tree in model.estimators_])
    rows, accumulated = [], 0.0
    for rank, position in enumerate(np.argsort(-native, kind="stable"), 1):
        accumulated += float(native[position])
        rows.append({"rank": rank, "feature": FEATURE_NAMES[position], "importance": float(native[position]),
                     "cumulative_importance": accumulated, "tree_std": float(forest[:, position].std()),
                     "tree_min": float(forest[:, position].min()), "tree_max": float(forest[:, position].max())})
    groups = []
    for group, names in GROUPS.items():
        columns = [FEATURE_INDEX[name] for name in names]
        groups.append({"group": group, "importance": float(native[columns].sum()),
                       "tree_std": float(forest[:, columns].sum(axis=1).std())})
    return rows, sorted(groups, key=lambda row: -row["importance"])


def permutation_importance(model, features: np.ndarray, regions: np.ndarray, references: dict) -> tuple[list, list, list]:
    """Executa cinco perturbações por item, reaproveitando cada probabilidade nos dois cortes."""
    runs, summaries = [], {"individual": [], "group": []}
    scopes = {"individual": {name: (name,) for name in FEATURE_NAMES}, "group": GROUPS}
    for scope, items in scopes.items():
        for name, members in items.items():
            current = []
            columns = tuple(FEATURE_INDEX[member] for member in members)
            for repetition in range(REPEATS):
                seed = SEED + repetition
                started = perf_counter()
                permuted = permute_columns(features, columns, seed)
                default, probabilities = forest_decisions(model, model.predict_proba(permuted))
                seconds = perf_counter() - started
                for cut, selected in (("default", default), ("d03", probabilities >= D03_THRESHOLD)):
                    scores = weighted_metrics(regions, selected)
                    changes = {f"delta_{metric}": scores[metric] - references[cut][metric] for metric in METRICS}
                    current.append({"scope": scope, "name": name, "cut": cut, "repetition": repetition + 1,
                                    "seed": seed, "seconds_shared_between_cuts": seconds, **scores, **changes})
            for cut in references:
                subset = [row for row in current if row["cut"] == cut]
                summary = {"name": name, "cut": cut, "repeats": REPEATS}
                for metric in METRICS:
                    deltas = [row[f"delta_{metric}"] for row in subset]
                    summary[f"mean_delta_{metric}"] = float(np.mean(deltas))
                    summary[f"std_delta_{metric}"] = float(np.std(deltas))
                summary["mean_IoU_drop"] = -summary["mean_delta_IoU"]
                summary["mean_F1_drop"] = -summary["mean_delta_F1"]
                summaries[scope].append(summary)
            runs.extend(current)
            print(f"Permutação D04: {scope} / {name}", flush=True)
    for rows in summaries.values():
        rows.sort(key=lambda row: (row["cut"], -row["mean_IoU_drop"]))
        for cut in references:
            for rank, row in enumerate((row for row in rows if row["cut"] == cut), 1):
                row["rank"] = rank
    return summaries["individual"], summaries["group"], runs


def outcome_groups(labels: np.ndarray, predicted: np.ndarray) -> dict:
    return {"TP": (labels == 1) & predicted, "TN": (labels == 0) & ~predicted,
            "FP": (labels == 0) & predicted, "FN": (labels == 1) & ~predicted,
            "ambiguous_positive": (labels == -1) & predicted,
            "ambiguous_negative": (labels == -1) & ~predicted}


def describe(features: np.ndarray, groups: dict) -> list[dict]:
    rows = []
    for name, mask in groups.items():
        selected = features[mask]
        quantiles = np.percentile(selected, [5, 25, 50, 75, 95], axis=0) if len(selected) else None
        averages = selected.mean(axis=0) if len(selected) else None
        deviations = selected.std(axis=0) if len(selected) else None
        for column, feature in enumerate(FEATURE_NAMES):
            values = ([float(averages[column]), float(deviations[column]), *map(float, quantiles[:, column])]
                      if quantiles is not None else [None] * 7)
            rows.append({"group": name, "feature": feature, "regions": len(selected),
                         **dict(zip(("mean", "std", "p05", "q1", "median", "q3", "p95"), values))})
    return rows


def compare_groups(features: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    """Usa delta de Cliff para comparar distribuições sem depender das unidades."""
    first, second = features[left], features[right]
    if not len(first) or not len(second):
        return {}
    scores = 2 * mannwhitneyu(first, second, axis=0, method="asymptotic").statistic / (len(first) * len(second)) - 1
    medians = np.median(first, axis=0) - np.median(second, axis=0)
    return {name: {"cliffs_delta": float(scores[i]), "median_difference": float(medians[i])}
            for i, name in enumerate(FEATURE_NAMES)}


def correlation_matrix(features: np.ndarray, *, spearman: bool = False) -> np.ndarray:
    """Calcula a correlação e deixa indefinidos os pares com uma coluna constante."""
    values = rankdata(features, axis=0) if spearman else features
    varying = np.ptp(values, axis=0) > 0
    matrix = np.full((features.shape[1], features.shape[1]), np.nan)
    if varying.any():
        matrix[np.ix_(varying, varying)] = np.corrcoef(values[:, varying], rowvar=False)
    return matrix


def correlation_rows(matrix: np.ndarray) -> list[dict]:
    return [{"feature": name, **{other: float(value) if np.isfinite(value) else None
                                for other, value in zip(FEATURE_NAMES, row)}}
            for name, row in zip(FEATURE_NAMES, matrix)]


def strong_pairs(matrix: np.ndarray) -> list[dict]:
    return sorted([{"first": FEATURE_NAMES[i], "second": FEATURE_NAMES[j], "r": float(matrix[i, j])}
                   for i, j in zip(*np.where(np.triu(np.abs(matrix) >= 0.8, k=1)))], key=lambda row: -abs(row["r"]))


def compare_i01(data: dict, groups: dict) -> list[dict]:
    regions, features = data["regions"], data["features"]
    rows = []
    for region_id, expected in I01_REGIONS.items():
        position, = np.flatnonzero((regions["image"] == I01) & (regions["id"] == region_id))
        category, = (name for name, mask in groups.items() if mask[position])
        if category != expected:
            raise ValueError(f"Rótulo de interesse divergente em I01 #{region_id}: {category}.")
        distribution = features[groups[category]]
        for index, feature in enumerate(FEATURE_NAMES):
            values = distribution[:, index]
            value = features[position, index]
            percentile = 100 * ((values < value).sum() + 0.5 * (values == value).sum()) / len(values)
            rows.append({"image_id": I01, "region_id": region_id, "outcome": category, "feature": feature,
                         "value": float(value), "outcome_regions": len(values), "percentile": float(percentile),
                         "outside_p05_p95": bool(percentile < 5 or percentile > 95)})
    return rows


def save_chart(figure, output: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=145)
    plt.close(figure)


def permutation_chart(output: Path, rows: list, filename: str) -> None:
    names = [row["name"] for row in rows if row["cut"] == "default"]
    figure, axis = plt.subplots(figsize=(9, max(4, len(names) * 0.38)))
    y = np.arange(len(names))
    for offset, cut in ((-0.18, "default"), (0.18, "d03")):
        lookup = {row["name"]: row for row in rows if row["cut"] == cut}
        axis.barh(y + offset, [100 * lookup[name]["mean_IoU_drop"] for name in names], height=0.33,
                  xerr=[100 * lookup[name]["std_delta_IoU"] for name in names], label=cut, capsize=2)
    axis.set(yticks=y, yticklabels=names, xlabel="Queda de IoU (p.p., média ± DP)")
    axis.axvline(0, color="gray", linewidth=0.8)
    axis.invert_yaxis()
    axis.legend()
    save_chart(figure, output, filename)


def boxplots(output: Path, features: np.ndarray, groups: dict, filename: str) -> None:
    figure, axes = plt.subplots(5, 3, figsize=(17, 15))
    nonempty = {name: mask for name, mask in groups.items() if mask.any()}
    labels = [f"{name}\nn={mask.sum():,}" for name, mask in nonempty.items()]
    for column, axis in enumerate(axes.flat):
        axis.boxplot([features[mask, column] for mask in nonempty.values()], showfliers=False)
        axis.set_xticks(range(1, len(labels) + 1), labels)
        axis.set_title(FEATURE_NAMES[column])
    save_chart(figure, output, filename)


def save_charts(output: Path, data: dict, outcomes: dict, classes: dict, native: list, individual: list,
                grouped: list, correlations: dict, i01: list) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = [row["feature"] for row in native]
    values = [row["importance"] for row in native]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(names[::-1], values[::-1])
    axis.set_xlabel("MDI normalizada")
    save_chart(figure, output, "native_importance")
    permutation_chart(output, individual, "individual_permutation")
    permutation_chart(output, grouped, "group_permutation")

    figure, axis = plt.subplots(figsize=(9, 5))
    lookup = {row["name"]: row for row in individual if row["cut"] == "default"}
    drops = [100 * lookup[name]["mean_IoU_drop"] for name in names]
    axis.scatter(values, drops)
    for value, drop, name in zip(values, drops, names):
        axis.annotate(name, (value, drop), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axis.set(xlabel="MDI normalizada", ylabel="Queda de IoU por permutação (p.p.)")
    save_chart(figure, output, "native_vs_permutation")

    for name, matrix in correlations.items():
        figure, axis = plt.subplots(figsize=(10, 9))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=70, ha="right")
        axis.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES)
        axis.set_title(f"{name.title()} · {len(data['features']):,} regiões")
        figure.colorbar(image, ax=axis)
        save_chart(figure, output, f"correlation_{name}")

    boxplots(output, data["features"], {key: outcomes[key] for key in ("TP", "TN", "FP", "FN")}, "outcomes")
    boxplots(output, data["features"], classes, "predominant_classes")
    for first, second in (("TP", "FN"), ("TN", "FP"), ("TP", "FP")):
        boxplots(output, data["features"], {key: outcomes[key] for key in (first, second)},
                 f"{first.lower()}_vs_{second.lower()}")

    figure, axis = plt.subplots(figsize=(11, 4))
    heatmap = np.array([row["percentile"] for row in i01]).reshape(len(I01_REGIONS), -1)
    image = axis.imshow(heatmap, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    axis.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=60, ha="right")
    axis.set_yticks(range(len(I01_REGIONS)), [f"#{i} · {group}" for i, group in I01_REGIONS.items()])
    figure.colorbar(image, ax=axis, label="Percentil dentro do grupo")
    save_chart(figure, output, "i01_feature_percentiles")


def run_experiment(dataset: Path, baseline: Path, d02: Path, d03: Path, output: Path,
                   *, reuse_features: bool = False) -> dict:
    started = perf_counter()
    ids = read_json(baseline / "split.json")["validation"]
    if len(ids) != 132 or len(set(ids)) != 132 or ids != read_json(d02 / "configuration.json")["validation_ids"]:
        raise ValueError("O manifesto e o D02 devem corresponder aos 132 IDs de validação.")
    artifact = joblib.load(baseline / "random_forest.joblib")
    model, config = artifact["model"], artifact["pipeline_config"]
    if (tuple(artifact["feature_names"]) != FEATURE_NAMES or model.n_features_in_ != len(FEATURE_NAMES)
            or config["features"]["quantization_levels"] != _LEVELS
            or config["labeling"]["class_id"] != DamageClass.STAIN):
        raise ValueError("A ordem das características ou a configuração do modelo é incompatível.")
    reference = read_regions(d02 / "region_statistics.csv")
    if len(reference) != 103621 or set(reference["image"]) != set(ids):
        raise ValueError("O D02 deve conter as 103.621 regiões da validação.")
    output.mkdir(parents=True, exist_ok=True)
    extraction_started = perf_counter()
    if reuse_features:
        with np.load(output / "validation_features.npz", allow_pickle=False) as saved:
            data = dict(saved.items())
        if (tuple(data["feature_names"]) != FEATURE_NAMES or
                not np.array_equal(np.sort(data["regions"], order=("image", "id")),
                                   np.sort(reference, order=("image", "id")))):
            raise ValueError("A matriz armazenada não corresponde à ordem das features ou ao D02.")
    else:
        data = extract_validation(dataset, output, ids, reference, config)
    extraction_seconds = 0.0 if reuse_features else perf_counter() - extraction_started
    features, regions = data["features"], data["regions"]
    if features.shape != (len(reference), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("Matriz de características incompatível ou com valores não finitos.")
    references, reproduction = verify_reference(model, features, regions, baseline, d03)
    print("Referências baseline e D03 verificadas", flush=True)

    analyses_started = perf_counter()
    native, native_groups = native_importance(model)
    permutation_started = perf_counter()
    individual, grouped, runs = permutation_importance(model, features, regions, references)
    permutation_seconds = perf_counter() - permutation_started
    outcomes = outcome_groups(data["training_labels"], regions["original"])
    classes = {name: 10 * data["class_pixel_counts"][:, damage] >= 7 * regions["area"]
               for damage, name in ((DamageClass.STAIN, "Stain"), (DamageClass.EROSION, "Erosion"),
                                    (DamageClass.BACKGROUND, "Background"))}
    comparisons = {f"{left}_vs_{right}": compare_groups(features, outcomes[left], outcomes[right])
                   for left, right in (("TP", "FN"), ("TN", "FP"), ("TP", "FP"))}
    class_comparisons = {f"{left}_vs_{right}": compare_groups(features, classes[left], classes[right])
                         for left, right in (("Erosion", "Stain"), ("Background", "Stain"), ("Erosion", "Background"))}
    correlations = {name: correlation_matrix(features, spearman=name == "spearman")
                    for name in ("pearson", "spearman")}
    i01 = compare_i01(data, outcomes)
    write_csv(output / "native_importance.csv", native)
    write_csv(output / "permutation_importance.csv", individual)
    write_csv(output / "group_permutation_importance.csv", grouped)
    write_csv(output / "permutation_runs.csv", runs)
    write_csv(output / "feature_statistics_by_outcome.csv", describe(features, outcomes))
    write_csv(output / "feature_statistics_by_class.csv", describe(features, classes))
    write_csv(output / "i01_feature_comparison.csv", i01)
    for name, matrix in correlations.items():
        write_csv(output / f"correlation_{name}.csv", correlation_rows(matrix))
    save_charts(output / "charts", data, outcomes, classes, native, individual, grouped, correlations, i01)
    write_json(output / "configuration.json", {"validation_ids": ids, "pipeline_config": config,
        "features": list(FEATURE_NAMES), "groups": GROUPS, "d02_csv": (d02 / "region_statistics.csv").as_posix(),
        "d03_threshold": D03_THRESHOLD, "seeds": list(range(SEED, SEED + REPEATS)), "predominance": PREDOMINANCE,
        "probability_verification_tolerance": PROBABILITY_TOLERANCE,
        "comparison_statistic": "Cliff's delta (first group minus second)"})
    report = {"images": len(ids), "regions": len(regions), "references": references, "reproduction": reproduction,
              "native_by_group": native_groups, "outcome_regions": {name: int(mask.sum()) for name, mask in outcomes.items()},
              "predominant_class_regions": {name: int(mask.sum()) for name, mask in classes.items()},
              "outcome_comparisons": comparisons, "predominant_class_comparisons": class_comparisons,
              "h1_expected_direction_fn_minus_tp": H1, "h2_expected_direction_fp_minus_tn": {k: -v for k, v in H1.items()},
              "strong_correlations": {name: strong_pairs(matrix) for name, matrix in correlations.items()},
              "extraction_and_npz_seconds": extraction_seconds, "permutation_seconds": permutation_seconds,
              "analyses_and_exports_seconds": perf_counter() - analyses_started, "wall_seconds": perf_counter() - started}
    write_json(output / "global_statistics.json", report)
    write_summary(output)
    return {key: report[key] for key in ("images", "regions", "reproduction", "extraction_and_npz_seconds",
                                         "permutation_seconds", "wall_seconds")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--baseline", type=Path, default=Path("results/region_full"))
    parser.add_argument("--d02", type=Path, default=Path("results/diagnostics/D02_false_positives"))
    parser.add_argument("--d03", type=Path, default=Path("results/diagnostics/D03_threshold_analysis"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/D04_feature_importance"))
    parser.add_argument("--reuse-features", action="store_true", help="Reutiliza o NPZ do D04 após validar seu alinhamento.")
    args = parser.parse_args()
    targets = tuple(path.name for path in args.output.iterdir()) if args.output.is_dir() else ()
    guard_cli_output(args.output, targets, owner="D04_feature_importance")
    print(run_experiment(args.dataset, args.baseline, args.d02, args.d03, args.output,
                         reuse_features=args.reuse_features), flush=True)


if __name__ == "__main__":
    main()
