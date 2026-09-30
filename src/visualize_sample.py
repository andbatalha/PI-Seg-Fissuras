from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


DATASET_PATH = Path("data/BFDD")

RGB_PATH = DATASET_PATH / "RGB"
LABEL_PATH = DATASET_PATH / "Label"
LABEL_COLOR_PATH = DATASET_PATH / "Label_color"


def main():

    # Escolhemos uma imagem do dataset
    rgb_files = list(RGB_PATH.glob("*.JPG"))

    rgb_file = random.choice(rgb_files)

    # Pegamos apenas o nome, sem extensão
    image_name = rgb_file.stem

    # Procuramos os arquivos correspondentes
    label_file = LABEL_PATH / f"{image_name}.png"
    label_color_file = LABEL_COLOR_PATH / f"{image_name}.png"

    # Carregamos as imagens
    rgb = cv2.imread(str(rgb_file))
    label = cv2.imread(
        str(label_file),
        cv2.IMREAD_UNCHANGED
    )
    label_color = cv2.imread(str(label_color_file))

    # OpenCV lê imagens coloridas como BGR.
    # Matplotlib espera RGB.
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    label_color = cv2.cvtColor(
        label_color,
        cv2.COLOR_BGR2RGB
    )

    print(f"Imagem: {image_name}")
    print(f"Valores da Label: {np.unique(label)}")

    # Criamos a janela
    plt.figure(figsize=(15, 5))

    # Imagem original
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    # Label numérica
    plt.subplot(1, 3, 2)
    plt.imshow(label)
    plt.title("Label")
    plt.axis("off")

    # Label colorida
    plt.subplot(1, 3, 3)
    plt.imshow(label_color)
    plt.title("Label Color")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()