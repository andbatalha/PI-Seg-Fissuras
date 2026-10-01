from pathlib import Path
import argparse
from dataset.classes import DamageClass

import cv2
import numpy as np

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


print("O programa iniciou!")

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
    Print the directory structure of the given root directory.

    Args:
        root (Path): The root directory to print the structure of.
        max_depth (int): The maximum depth to print.
    """

    print("\n=== Estrutura do Diretorio ===")

    for path in root.rglob("*"):

        relative_path = path.relative_to(root)

        depth = len(relative_path.parts)

        if depth > max_depth:
            continue

        indentation = "  " * (depth - 1)

        if path.is_dir():
            print(f"{indentation}[DIR] {path.name}")


def inspect_images(images: list[Path], limit: int = 10) -> None:
    """
    Abre algumas imagens e mostra informações sobre elas.
    """

    print("\n=== Imagens encontradas ===\n")

    print(f"Total de imagens: {len(images)}")

    if len(images) == 0:
        print("Nenhuma imagem encontrada.")
        return

    for path in images[:limit]:

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Falha ao abrir a imagem: {path}")
            continue

        print(f"Arquivo: {path}")
        print(f"Dimensões: {image.shape}")
        print(f"Tipo dos dados: {image.dtype}")
        print("-" * 60)


def count_images_by_folder(root: Path) -> None:
    """
    Conta quantos arquivos de imagem existem em cada
    pasta principal do dataset.
    """

    print("\n=== Quantidade de imagens por pasta ===\n")

    for folder in sorted(root.iterdir()):

        if not folder.is_dir():
            continue

        images = find_images(folder)

        print(f"{folder.name}: {len(images)}")


def inspect_folders(root: Path) -> None:
    """
    Abre a primeira imagem encontrada em cada pasta principal
    e mostra suas características.
    """

    print("\n=== Exemplo de cada pasta ===\n")

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

        print(f"Pasta: {folder.name}")
        print(f"Arquivo: {path.name}")
        print(f"Dimensões: {image.shape}")
        print(f"Tipo: {image.dtype}")
        print("-" * 60)


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

    print("\n=== Investigação das Labels ===\n")

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

    print(f"Quantidade de Labels analisadas: {len(images)}")

    print(
        f"Valores encontrados no dataset: "
        f"{sorted(all_values)}"
    )

    print("\nPresença de cada valor nas imagens:\n")

    for value in sorted(class_occurrences):
        print(
            f"Valor {value} ({DamageClass(value).name}): "
            f"{class_occurrences[value]} imagens"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspeciona a estrutura do dataset BFDD")

    parser.add_argument(
        "dataset_path",
        type=str,
        help="Caminho para a pasta raiz do dataset BFDD"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"O caminho informado não existe ou não é uma pasta: {dataset_root}")

    print(f"\nDataset encontrado em:")
    print(dataset_root.resolve())

    print_directory_structure(dataset_root)

    count_images_by_folder(dataset_root)

    images = find_images(dataset_root)

    inspect_images(images)

    inspect_folders(dataset_root)

    inspect_label_values(dataset_root)

if __name__ == "__main__":
    main()