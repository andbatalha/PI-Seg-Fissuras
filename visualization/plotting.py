from pathlib import Path
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def pixel_error_map(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Codifica TN, FP, FN e TP em cinza, laranja, roxo e verde."""
    return _ERROR_PALETTE[2 * reference + prediction]


_ERROR_PALETTE = np.array([[76, 76, 76], [229, 126, 34], [112, 74, 168], [46, 153, 79]], dtype=np.uint8)


def pixel_error_legend():
    """Retorna a legenda da paleta usada nas figuras de avaliação."""
    names = ("Verdadeiro negativo", "Falso positivo", "Falso negativo", "Verdadeiro positivo")
    return [Patch(facecolor=color / 255, label=name) for color, name in zip(_ERROR_PALETTE, names)]


def comparison_panels(images_and_titles, *, columns: int | None = None):
    """Dispõe imagens em painéis, usando escala fixa para máscaras binárias."""
    columns = columns or len(images_and_titles)
    rows = ceil(len(images_and_titles) / columns)
    figure, grid = plt.subplots(rows, columns, figsize=(20, 5 * rows), squeeze=False)
    axes = grid.ravel()[:len(images_and_titles)]
    for unused in grid.ravel()[len(axes):]:
        unused.remove()
    for axis, (image, title) in zip(axes, images_and_titles):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    return figure, axes


def save_comparison(figure, output: Path) -> None:
    """Ajusta o espaço dos painéis e grava o PNG no caminho indicado."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
