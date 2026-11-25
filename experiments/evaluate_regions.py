import argparse
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np

from experiments.evaluate_thresholds import apply_threshold, optimal_threshold
from experiments.reporting import guard_cli_output, write_summary

from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.dataset.masks import create_binary_mask
from src.evaluation import count_binary_pixels, count_region_pixels, pixel_metrics
from src.region_labeling import label_regions
from src.region_prediction import model_feature_groups, predict_regions, reconstruct_binary_mask
from visualization.plotting import comparison_panels, pixel_error_legend, pixel_error_map, save_comparison


def visualize_prediction(rgb, reference, prediction, output: Path) -> None:
    """Grava quatro painéis com as categorias de erro e uma legenda."""
    figure, axes = comparison_panels([
        (rgb, "Imagem original"),
        (reference, "Máscara dos autores"),
        (prediction, "Máscara prevista"),
        (pixel_error_map(reference, prediction), "Mapa de erros"),
    ])
    axes[-1].legend(handles=pixel_error_legend(), loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2)
    save_comparison(figure, output)
    plt.close(figure)


def infer_image(artifact: dict, dataset: Path, image_id: str, *,
                threshold: float | None = None, details: bool = False) -> tuple:
    """Abre a anotação somente após a inferência e devolve os dados da imagem."""
    rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
    started = perf_counter()
    prediction = (predict_regions(artifact, rgb, include_details=True) if details
                  else predict_regions(artifact, rgb, threshold=threshold))
    inference_seconds = perf_counter() - started
    label = load_label(dataset / "Label" / f"{image_id}.png")

    return rgb, label, prediction, inference_seconds


def image_record(image_id: str, counts: dict, inference_seconds: float) -> dict:
    return {
        "image_id": image_id,
        "pixels": sum(counts.values()),
        "has_stain": bool(counts["TP"] + counts["FN"]),
        "counts": counts,
        "metrics": pixel_metrics(counts),
        "inference_seconds": inference_seconds,
    }


def evaluate_image(artifact: dict, dataset: Path, image_id: str, *,
                   threshold: float | None = None) -> tuple[dict, tuple]:
    """Avalia por comparação direta da referência com a máscara prevista."""
    rgb, label, prediction, seconds = infer_image(artifact, dataset, image_id, threshold=threshold)
    reference = create_binary_mask(label, DamageClass.STAIN)
    counts = count_binary_pixels(reference, prediction.mask)
    return image_record(image_id, counts, seconds), (rgb, reference, prediction.mask)


def evaluation_configuration(artifact: dict, model_path: Path | None, threshold: float | None,
                             *, optimize: bool = False) -> dict:
    """Identifica o modelo e a regra de decisão no relatório da execução."""
    return {
        **({"model": model_path.as_posix()} if model_path else {}),
        "feature_groups": list(model_feature_groups(artifact)),
        "feature_count": len(artifact["feature_names"]),
        "pipeline_config": artifact["pipeline_config"],
        "decision": "max_iou_validation" if optimize else "default" if threshold is None else "threshold",
        "threshold": threshold,
    }


def stain_iou_summary(records: list[dict]) -> dict:
    ious = [record["metrics"]["IoU"] for record in records if record["has_stain"]]
    return {"mean_iou_with_stain": mean(ious) if ious else None,
            "median_iou_with_stain": median(ious) if ious else None}


def evaluate_validation(artifact: dict, dataset: Path, image_ids: list[str], output: Path, *,
                        threshold: float | None = None, optimize: bool = False,
                        model_path: Path | None = None) -> None:
    """Avalia os IDs registrados e grava resultados globais e individuais em JSON."""
    if optimize and threshold is not None:
        raise ValueError("Escolha limiar personalizado ou busca do limiar ótimo.")
    started = perf_counter()
    configuration = None
    if model_path is not None or threshold is not None or optimize:
        configuration = evaluation_configuration(artifact, model_path, threshold, optimize=optimize)
    totals = dict.fromkeys(("TP", "TN", "FP", "FN"), 0)
    records = []
    errors = []
    examples = []
    pending = []
    remaining_examples = {True: 3, False: 2}
    for index, image_id in enumerate(image_ids, start=1):
        try:
            if optimize:
                rgb, label, result, seconds = infer_image(artifact, dataset, image_id, details=True)
                annotated = label_regions(result.regions, label, **artifact["pipeline_config"]["labeling"])
                if not np.array_equal(annotated.region_ids, result.region_ids):
                    raise ValueError("Os IDs das regiões previstas e anotadas divergem.")
                rows = np.rec.fromarrays(
                    (result.probabilities, annotated.target_counts, annotated.pixel_counts - annotated.target_counts),
                    names=("probability", "stain", "other"),
                )
                has_stain = bool(annotated.target_counts.any())
                spatial = None
                if remaining_examples[has_stain]:
                    remaining_examples[has_stain] -= 1
                    spatial = (rgb, create_binary_mask(label, DamageClass.STAIN), result.regions, result.region_ids)
                pending.append((image_id, rows, seconds, spatial))
            else:
                record, panels = evaluate_image(artifact, dataset, image_id, threshold=threshold)
                records.append(record)
                has_stain = record["has_stain"]
                if remaining_examples[has_stain]:
                    remaining_examples[has_stain] -= 1
                    path = output / "pixel_examples" / f"{image_id}.png"
                    visualize_prediction(*panels, path)
                    examples.append({"image_id": image_id, "has_stain": has_stain, "png": path.as_posix()})
        except (OSError, ValueError) as error:
            errors.append({"image_id": image_id, "error": str(error)})
            print(f"Erro em {image_id}: {error}", flush=True)
            continue
        if index == 1 or index % 10 == 0 or index == len(image_ids):
            print(f"Validação por pixel | {index}/{len(image_ids)} imagens", flush=True)

    if optimize and pending and not errors:
        threshold, _, _ = optimal_threshold(np.concatenate([rows for _, rows, _, _ in pending]))
        for image_id, rows, seconds, spatial in pending:
            selected = apply_threshold(rows["probability"], threshold)
            counts = count_region_pixels(rows["stain"], rows["other"], selected)
            records.append(image_record(image_id, counts, seconds))
            if spatial is not None:
                rgb, reference, regions, ids = spatial
                mask = reconstruct_binary_mask(regions, ids, selected)
                path = output / "pixel_examples" / f"{image_id}.png"
                visualize_prediction(rgb, reference, mask, path)
                examples.append({"image_id": image_id, "has_stain": bool(reference.any()), "png": path.as_posix()})

    for name in totals:
        totals[name] = sum(record["counts"][name] for record in records)
    seconds = sum(record["inference_seconds"] for record in records)
    report = {
        "images": len(records),
        "pixels": sum(totals.values()),
        "counts": totals,
        "confusion_matrix_rows_true_columns_predicted_0_1": [
            [totals["TN"], totals["FP"]], [totals["FN"], totals["TP"]],
        ],
        "metrics": pixel_metrics(totals) if records else None,
        **stain_iou_summary(records),
        "processing_seconds": perf_counter() - started,
        "inference_seconds": seconds,
        "seconds_per_image": seconds / len(records) if records else None,
        "image_errors": len(errors),
        "errors": errors,
        "examples": examples,
    }
    if configuration is not None:
        report["configuration"] = {**configuration, "threshold": threshold}
    summary = json.dumps(report, indent=2, ensure_ascii=False)
    (output / "pixel_validation_metrics.json").write_text(summary + "\n", encoding="utf-8")
    individual = json.dumps(records, indent=2, ensure_ascii=False)
    (output / "pixel_validation_per_image.json").write_text(individual + "\n", encoding="utf-8")
    write_summary(output)
    print(summary)
    if errors:
        raise SystemExit("A avaliação está incompleta. Consulte os erros registrados no relatório.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação por pixel da segmentação de manchas.")
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--output", type=Path, default=Path("results/region_full"))
    parser.add_argument("--source", type=Path, help="Diretório do modelo e do manifesto quando diferente da saída.")
    parser.add_argument("--image-id", help="Avalia e salva o PNG apenas deste ID de validação.")
    args = parser.parse_args()
    source = args.source or args.output
    manifest = json.loads((source / "split.json").read_text(encoding="utf-8"))
    image_ids = manifest["validation"]
    if len(image_ids) != 132 or len(set(image_ids)) != 132:
        parser.error("O manifesto deve conter os 132 IDs distintos da validação completa.")
    if args.image_id and args.image_id not in image_ids:
        parser.error("O ID solicitado não pertence à validação registrada.")
    if args.image_id:
        guard_cli_output(args.output, (f"pixel_visualizations/{args.image_id}.png",
                                       f"pixel_visualizations/{args.image_id}"))
    else:
        guard_cli_output(args.output, ("pixel_validation_metrics.json", "pixel_validation_per_image.json",
                                       "pixel_examples", "summary.md"))
    artifact = joblib.load(source / "random_forest.joblib")
    if artifact["pipeline_config"]["labeling"]["class_id"] != DamageClass.STAIN:
        parser.error("O modelo salvo deve ter sido treinado para Stain.")

    if args.image_id:
        record, panels = evaluate_image(artifact, args.dataset, args.image_id)
        path = args.output / "pixel_visualizations" / f"{args.image_id}.png"
        visualize_prediction(*panels, path)
        report = path.parent / args.image_id
        report.mkdir(exist_ok=True)
        (report / "individual_metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        write_summary(report)
        print(json.dumps(record, indent=2))
        print(f"PNG: {path}")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        evaluate_validation(artifact, args.dataset, image_ids, args.output)


if __name__ == "__main__":
    main()
