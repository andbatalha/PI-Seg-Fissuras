import argparse
import json
from pathlib import Path

from src.dataset.classes import DamageClass
from src.dataset.index import build_dataset_index
from src.dataset.split import split_dataset, split_train_validation
from src.region_dataset import build_region_dataset
from src.region_training import train_region_forest


def select_smoke_images(image_ids: list[str], dataset_index: dict) -> list[str]:
    """Seleciona uma amostra pequena com e sem mancha para verificar a integração."""
    with_stain = []
    without_stain = []
    for name in image_ids:
        if DamageClass.STAIN in dataset_index[name]:
            with_stain.append(name)
        else:
            without_stain.append(name)
    return with_stain[:3] + without_stain[:2]


def save_report(path: Path, report: dict) -> None:
    """Registra os resultados em JSON e os apresenta no terminal."""
    content = json.dumps(report, indent=2, ensure_ascii=False)
    path.write_text(content + "\n", encoding="utf-8")
    print(content, flush=True)


def build_datasets(dataset_root: Path, output: Path, smoke: bool) -> None:
    """Reutiliza a divisão do projeto e materializa os dois conjuntos de regiões."""
    dataset_index = dict(sorted(build_dataset_index(dataset_root / "Label").items()))
    development, test = split_dataset(dataset_index)
    seed = 42
    validation_ratio = 0.2
    train, validation = split_train_validation(
        development, dataset_index, validation_ratio=validation_ratio, seed=seed
    )
    if (len(set(train + validation + test)) != len(train) + len(validation) + len(test)
            or set(train + validation + test) != set(dataset_index)):
        raise ValueError("A divisão contém IDs duplicados ou não cobre todo o BFDD.")

    if smoke:
        train = select_smoke_images(train, dataset_index)
        validation = select_smoke_images(validation, dataset_index)

    manifest = {
        "strategy": "Divisão existente, estratificada por combinações de classes",
        "validation_ratio": validation_ratio,
        "seed": seed,
        "smoke": smoke,
        "train": train,
        "validation": validation,
        "test_reserved": test,
    }
    (output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Divisão | treino {len(train)} | validação {len(validation)} | teste reservado {len(test)}")
    statistics = {
        "training": build_region_dataset(dataset_root, train, output / "train.npz"),
        "validation": build_region_dataset(dataset_root, validation, output / "validation.npz"),
    }
    save_report(output / "dataset_statistics.json", statistics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset e Random Forest de regiões de mancha.")
    parser.add_argument("stage", choices=("build", "train"))
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Processa até cinco imagens por conjunto.")
    args = parser.parse_args()
    if args.smoke and args.stage != "build":
        parser.error("--smoke se aplica à etapa build.")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        build_datasets(args.dataset, args.output, args.smoke)
    else:
        report = train_region_forest(
            args.output / "train.npz", args.output / "validation.npz",
            args.output / "random_forest.joblib",
        )
        save_report(args.output / "validation_metrics.json", report)


if __name__ == "__main__":
    main()
