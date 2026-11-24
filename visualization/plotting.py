from pathlib import Path

import matplotlib.pyplot as plt


def comparison_panels(images_and_titles):
    """Dispõe imagens em uma linha, usando escala fixa para máscaras binárias."""
    figure, axes = plt.subplots(1, len(images_and_titles), figsize=(20, 5))
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
