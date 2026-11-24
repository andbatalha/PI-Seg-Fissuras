"""Inspeciona pares RGB e Label selecionados no manifesto do BFDD."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.dataset.index import get_present_classes
from src.dataset.loader import load_image, load_label


def inspect_images(dataset: Path, image_ids: list[str]) -> None:
    """Exibe formatos e presença de classes nos pares selecionados."""
    formats = Counter()
    class_occurrences = Counter()
    print(f"BFDD | {dataset.resolve()} | {len(image_ids)} pares selecionados")
    for image_id in image_ids:
        rgb = load_image(dataset / "RGB" / f"{image_id}.JPG")
        label = load_label(dataset / "Label" / f"{image_id}.png")
        if rgb.shape[:2] != label.shape:
            raise ValueError(f"RGB e Label têm dimensões incompatíveis para {image_id}.")
        description = f"RGB {rgb.shape} {rgb.dtype} | Label {label.shape} {label.dtype}"
        formats[description] += 1
        class_occurrences.update(get_present_classes(label))
        print(f"{image_id} | {description}")

    print("\nFormatos por quantidade de pares")
    for description, count in formats.items():
        print(f"{count:>4} | {description}")
    print("\nClasses por quantidade de imagens")
    for damage_class, count in sorted(class_occurrences.items()):
        print(f"{damage_class.name:<15} {count:>4} ({count / len(image_ids):.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/BFDD"))
    parser.add_argument("--manifest", type=Path, default=Path("results/region_full/split.json"))
    parser.add_argument("--subset", choices=("train", "validation"), default="validation")
    parser.add_argument("--image-id", help="Inspeciona apenas este ID do subconjunto selecionado.")
    parser.add_argument("--limit", type=int, default=5, help="Total de pares a inspecionar.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit deve ser positivo.")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        authorized_ids = manifest[args.subset]
        if args.image_id and args.image_id not in authorized_ids:
            parser.error("O ID não pertence ao subconjunto selecionado.")
        selected = [args.image_id] if args.image_id else authorized_ids[:args.limit]
        inspect_images(args.dataset, selected)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
