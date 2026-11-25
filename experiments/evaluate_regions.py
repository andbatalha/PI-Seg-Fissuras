import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.dataset.masks import create_binary_mask
from src.evaluation import count_binary_pixels, pixel_metrics
from src.region_prediction import predict_regions
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


def evaluate_image(artifact: dict, dataset: Path, image_id: str) -> tuple[dict, tuple]:
    """Executa a predição antes de abrir a anotação usada para avaliar a imagem."""
    rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
    started = perf_counter()
    prediction = predict_regions(artifact, rgb).mask
    inference_seconds = perf_counter() - started
    label = load_label(dataset / "Label" / f"{image_id}.png")
    reference = create_binary_mask(label, DamageClass.STAIN)
    counts = count_binary_pixels(reference, prediction)
    record = {
        "image_id": image_id,
        "pixels": reference.size,
        "has_stain": bool(reference.any()),
        "counts": counts,
        "metrics": pixel_metrics(counts),
        "inference_seconds": inference_seconds,
    }
    return record, (rgb, reference, prediction)


def evaluate_validation(artifact: dict, dataset: Path, image_ids: list[str], output: Path) -> None:
    """Avalia os IDs registrados e grava resultados globais e individuais em JSON."""
    totals = dict.fromkeys(("TP", "TN", "FP", "FN"), 0)
    records = []
    errors = []
    examples = []
    remaining_examples = {True: 3, False: 2}
    for index, image_id in enumerate(image_ids, start=1):
        try:
            record, panels = evaluate_image(artifact, dataset, image_id)
        except (OSError, ValueError) as error:
            errors.append({"image_id": image_id, "error": str(error)})
            print(f"Erro em {image_id}: {error}", flush=True)
            continue
        records.append(record)
        for name in totals:
            totals[name] += record["counts"][name]

        # A ordem do manifesto e a presença na referência determinam os exemplos.
        has_stain = record["has_stain"]
        if remaining_examples[has_stain]:
            path = output / "pixel_examples" / f"{image_id}.png"
            visualize_prediction(*panels, path)
            examples.append({"image_id": image_id, "has_stain": has_stain, "png": path.as_posix()})
            remaining_examples[has_stain] -= 1
        if index == 1 or index % 10 == 0 or index == len(image_ids):
            print(f"Validação por pixel | {index}/{len(image_ids)} imagens", flush=True)

    seconds = sum(record["inference_seconds"] for record in records)
    report = {
        "images": len(records),
        "pixels": sum(totals.values()),
        "counts": totals,
        "confusion_matrix_rows_true_columns_predicted_0_1": [
            [totals["TN"], totals["FP"]], [totals["FN"], totals["TP"]],
        ],
        "metrics": pixel_metrics(totals) if records else None,
        "inference_seconds": seconds,
        "seconds_per_image": seconds / len(records) if records else None,
        "image_errors": len(errors),
        "errors": errors,
        "examples": examples,
    }
    summary = json.dumps(report, indent=2, ensure_ascii=False)
    (output / "pixel_validation_metrics.json").write_text(summary + "\n", encoding="utf-8")
    individual = json.dumps(records, indent=2, ensure_ascii=False)
    (output / "pixel_validation_per_image.json").write_text(individual + "\n", encoding="utf-8")
    print(summary)
    if errors:
        raise SystemExit("A avaliação está incompleta. Consulte os erros registrados no relatório.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação por pixel da segmentação de manchas.")
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--output", type=Path, default=Path("results/region_full"))
    parser.add_argument("--image-id", help="Avalia e salva o PNG apenas deste ID de validação.")
    args = parser.parse_args()
    manifest = json.loads((args.output / "split.json").read_text(encoding="utf-8"))
    image_ids = manifest["validation"]
    if len(image_ids) != 132 or len(set(image_ids)) != 132:
        parser.error("O manifesto deve conter os 132 IDs distintos da validação completa.")
    if args.image_id and args.image_id not in image_ids:
        parser.error("O ID solicitado não pertence à validação registrada.")
    artifact = joblib.load(args.output / "random_forest.joblib")
    if artifact["pipeline_config"]["labeling"]["class_id"] != DamageClass.STAIN:
        parser.error("O modelo salvo deve ter sido treinado para Stain.")

    if args.image_id:
        record, panels = evaluate_image(artifact, args.dataset, args.image_id)
        path = args.output / "pixel_visualizations" / f"{args.image_id}.png"
        visualize_prediction(*panels, path)
        print(json.dumps(record, indent=2))
        print(f"PNG: {path}")
    else:
        evaluate_validation(artifact, args.dataset, image_ids, args.output)


if __name__ == "__main__":
    main()
