import argparse
import json
from pathlib import Path

from experiments.reporting import guard_cli_output, read_report, write_summary
from src.dataset.classes import DamageClass
from src.dataset.index import get_present_classes
from src.dataset.loader import load_label
from src.feature_extractor import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS, canonical_feature_groups
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


def build_datasets(dataset_root: Path, output: Path, smoke: bool, *,
                   manifest_path: Path = Path("results/region_full/split.json"),
                   feature_groups=DEFAULT_FEATURE_GROUPS) -> None:
    """Reutiliza a divisão do projeto e materializa os dois conjuntos de regiões."""
    manifest = read_report(manifest_path)
    groups = canonical_feature_groups(feature_groups)
    train, validation = manifest["train"], manifest["validation"]
    if (not train or not validation or len(set(train + validation)) != len(train) + len(validation)
            or set(train + validation).intersection(manifest.get("test_reserved", ()))):
        raise ValueError("O manifesto contém subconjuntos vazios ou sobrepostos.")

    if smoke:
        dataset_index = {name: get_present_classes(load_label(dataset_root / "Label" / f"{name}.png"))
                         for name in train + validation}
        train = select_smoke_images(train, dataset_index)
        validation = select_smoke_images(validation, dataset_index)
        manifest = {**manifest, "smoke": True, "train": train, "validation": validation}

    output.mkdir(parents=True, exist_ok=True)
    print(f"Divisão registrada | treino {len(train)} | validação {len(validation)}")
    statistics = {
        "configuration": {"feature_groups": list(groups),
                          "feature_count": sum(len(FEATURE_GROUPS[group]) for group in groups)},
        "training": build_region_dataset(dataset_root, train, output / "train.npz", feature_groups=groups),
        "validation": build_region_dataset(dataset_root, validation, output / "validation.npz", feature_groups=groups),
    }
    save_report(output / "dataset_statistics.json", statistics)
    (output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_summary(output)


def train_from_datasets(source: Path, output: Path, *, manifest_path: Path) -> None:
    """Treina a partir de NPZ salvos e registra o manifesto quando disponível."""
    output.mkdir(parents=True, exist_ok=True)
    report = train_region_forest(source / "train.npz", source / "validation.npz", output / "random_forest.joblib")
    save_report(output / "validation_metrics.json", report)
    destination = output / "split.json"
    if not destination.exists() and manifest_path.is_file():
        destination.write_bytes(manifest_path.read_bytes())
    write_summary(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset e Random Forest de regiões de mancha.")
    parser.add_argument("stage", choices=("build", "train"))
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Processa até cinco imagens por conjunto.")
    parser.add_argument("--manifest", type=Path, default=Path("results/region_full/split.json"))
    parser.add_argument("--source", type=Path, help="Diretório dos NPZ para treino, se diferente da saída.")
    parser.add_argument("--feature-groups", nargs="+", choices=tuple(FEATURE_GROUPS),
                        help="Grupos de características para a etapa build.")
    args = parser.parse_args()
    if args.smoke and args.stage != "build":
        parser.error("--smoke se aplica à etapa build.")
    if args.feature_groups and args.stage != "build":
        parser.error("--feature-groups se aplica à etapa build.")

    if args.stage == "build":
        guard_cli_output(args.output, ("train.npz", "validation.npz", "split.json", "dataset_statistics.json", "summary.md"))
        build_datasets(args.dataset, args.output, args.smoke, manifest_path=args.manifest,
                       feature_groups=args.feature_groups or DEFAULT_FEATURE_GROUPS)
    else:
        source = args.source or args.output
        manifest_copy = not (args.output / "split.json").exists() and (source / "split.json").is_file()
        targets = ("random_forest.joblib", "validation_metrics.json", "summary.md")
        guard_cli_output(args.output, (*targets, "split.json") if manifest_copy else targets)
        train_from_datasets(source, args.output, manifest_path=source / "split.json")


if __name__ == "__main__":
    main()
