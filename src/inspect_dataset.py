from pathlib import Path

import cv2
import numpy as np

from dataset.classes import DamageClass


DATASET_PATH = Path("data/BFDD")

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif"
]


def find_images(root: Path) -> list[Path]:
    """
    Procura arquivos de imagem dentro da pasta informada
    e também dentro de todas as suas subpastas.
    """

    images = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


def print_directory_structure(root: Path, max_depth: int = 3) -> None:
    """
    Exibe a estrutura de diretórios do dataset.
    """

    print("\n" + "=" * 60)
    print("ESTRUTURA DO DATASET")
    print("=" * 60)

    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        depth = len(relative_path.parts)

        if depth > max_depth:
            continue

        indentation = "  " * (depth - 1)

        if path.is_dir():
            print(f"{indentation}└── {path.name}/")


def count_images_by_folder(root: Path) -> None:
    """
    Conta quantos arquivos de imagem existem em cada
    pasta principal do dataset.
    """

    print("\n" + "=" * 60)
    print("QUANTIDADE DE IMAGENS POR PASTA")
    print("=" * 60)

    total = 0

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        images = find_images(folder)
        amount = len(images)

        print(f"{folder.name:<40} {amount:>5}")

        total += amount

    print("-" * 60)
    print(f"{'TOTAL':<40} {total:>5}")


def inspect_folders(root: Path) -> None:
    """
    Exibe as características de uma imagem de exemplo
    de cada pasta principal.
    """

    print("\n" + "=" * 60)
    print("CARACTERÍSTICAS DAS IMAGENS")
    print("=" * 60)

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        images = find_images(folder)

        if len(images) == 0:
            continue

        path = images[0]

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            print(f"Não foi possível abrir: {path}")
            continue

        print(f"\nPasta:     {folder.name}")
        print(f"Exemplo:   {path.name}")
        print(f"Dimensões: {image.shape}")
        print(f"Tipo:      {image.dtype}")


def inspect_label_values(root: Path) -> None:
    """
    Investiga os valores de classe presentes em todas
    as máscaras da pasta Label.
    """

    label_folder = root / "Label"
    images = find_images(label_folder)

    if len(images) == 0:
        print("Nenhuma Label encontrada.")
        return

    all_values = set()
    class_occurrences = {}

    for path in images:
        label = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if label is None:
            print(f"Não foi possível abrir: {path}")
            continue

        unique_values = np.unique(label)

        for value in unique_values:
            value = int(value)

            all_values.add(value)

            if value not in class_occurrences:
                class_occurrences[value] = 0

            class_occurrences[value] += 1

    print("\n" + "=" * 60)
    print("CLASSES DAS LABELS")
    print("=" * 60)

    print(f"\nLabels analisadas: {len(images)}")
    print(f"Valores encontrados: {sorted(all_values)}")

    print("\nPresença das classes:")
    print("-" * 60)

    for value in sorted(class_occurrences):
        damage_class = DamageClass(value)
        occurrences = class_occurrences[value]
        percentage = occurrences / len(images) * 100

        print(
            f"{value} - {damage_class.name:<15} "
            f"{occurrences:>4} imagens "
            f"({percentage:>5.1f}%)"
        )


def main():
    dataset_root = DATASET_PATH

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(
            f"O dataset não foi encontrado em: {dataset_root.resolve()}"
        )

    print("\n" + "=" * 60)
    print("INSPEÇÃO DO DATASET BFDD")
    print("=" * 60)

    print(f"\nCaminho: {dataset_root.resolve()}")

    print_directory_structure(dataset_root)
    count_images_by_folder(dataset_root)
    inspect_folders(dataset_root)
    inspect_label_values(dataset_root)

    print("\n" + "=" * 60)
    print("INSPEÇÃO CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    main()
