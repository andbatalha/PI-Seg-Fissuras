
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import slic, mark_boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--class-id", type=int, default=5)
    parser.add_argument("--segments", type=int, default=200)
    parser.add_argument("--compactness", type=float, default=10)
    parser.add_argument(
        "--output",
        default="outputs/diagnostico_slic.png"
    )
    args = parser.parse_args()

    # Imagem original e label dos autores
    rgb = np.asarray(
        Image.open(args.image).convert("RGB")
    )
    gt = np.asarray(Image.open(args.label))

    if gt.ndim != 2:
        raise ValueError(
            "Use a label de índices (Label), "
            "não Label_color."
        )

    if gt.shape != rgb.shape[:2]:
        raise ValueError("Imagem e label incompatíveis.")

    # Segmentação automática, sem usar a label
    regions = slic(
        rgb,
        n_segments=args.segments,
        compactness=args.compactness,
        sigma=1,
        start_label=1,
        channel_axis=-1
    )

    # Ground truth binário da classe escolhida
    target_mask = gt == args.class_id

    # Análise da sobreposição de cada região
    total = np.bincount(regions.ravel())
    target = np.bincount(
        regions.ravel(),
        weights=target_mask.ravel(),
        minlength=len(total)
    )

    valid = total > 0
    fractions = target[valid] / total[valid]

    print(f"Regiões geradas: {valid.sum()}")
    print(f"Regiões com algum dano: {(fractions > 0).sum()}")
    print(f"Regiões >= 70% dano: {(fractions >= 0.7).sum()}")
    print(
        "Regiões mistas (10%-70%):",
        ((fractions > 0.1) & (fractions < 0.7)).sum()
    )

    # Visualização comparativa
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Imagem original")

    axes[1].imshow(target_mask, cmap="gray")
    axes[1].set_title("Ground truth da classe")

    axes[2].imshow(
        mark_boundaries(rgb / 255.0, regions)
    )
    axes[2].set_title("Segmentação SLIC")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
