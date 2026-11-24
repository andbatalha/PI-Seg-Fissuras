
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

from src.dataset.classes import DamageClass
from src.dataset.masks import create_binary_mask
from src.region_labeling import label_regions
from src.segmentation import segment_regions
from visualization.plotting import comparison_panels, save_comparison


def maximum_region_iou(region_sizes, target_counts):
    """Calcula a maior IoU ao selecionar apenas regiões inteiras e não vazias."""
    target_pixels = target_counts.sum()
    if target_pixels == 0:
        return None

    fractions = target_counts / region_sizes
    order = np.argsort(-fractions)
    true_positives = np.cumsum(target_counts[order])
    false_positives = np.cumsum((region_sizes - target_counts)[order])

    # A união é formada pelos pixels reais da classe e pelos falsos positivos.
    ious = true_positives / (target_pixels + false_positives)
    return float(ious.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--class-id", type=int, default=5)
    parser.add_argument("--negative-max", type=float, default=0.10)
    parser.add_argument("--positive-min", type=float, default=0.70)
    parser.add_argument("--segments", type=int)
    parser.add_argument("--compactness", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--start-label", type=int, choices=(0, 1))
    parser.add_argument(
        "--output",
        default="results/diagnostics/diagnostico_slic.png"
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
    slic_options = {
        "n_segments": args.segments,
        "compactness": args.compactness,
        "sigma": args.sigma,
        "start_label": args.start_label,
    }
    regions = segment_regions(
        rgb, **{name: value for name, value in slic_options.items() if value is not None}
    )

    # Ground truth binário da classe escolhida
    target_mask = create_binary_mask(gt, DamageClass(args.class_id))
    labeling = label_regions(
        regions, gt, class_id=args.class_id,
        negative_max=args.negative_max, positive_min=args.positive_min,
    )
    fractions = labeling.target_fractions

    print(f"Regiões geradas: {labeling.region_ids.size}")
    print(f"Regiões com algum dano: {(fractions > 0).sum()}")
    print(f"Regiões >= 70% dano: {(fractions >= 0.7).sum()}")
    print(
        "Regiões mistas (10%-70%):",
        ((fractions > 0.1) & (fractions < 0.7)).sum()
    )
    max_iou = maximum_region_iou(labeling.pixel_counts, labeling.target_counts)
    if max_iou is None:
        print(f"IoU máxima teórica: indefinida (sem pixels da classe {args.class_id})")
    else:
        print(f"IoU máxima teórica: {max_iou:.2%}")

    categories = (
        (1, "Positivas", (0.18, 0.60, 0.31)),
        (0, "Negativas", (0.50, 0.50, 0.50)),
        (-1, "Ambíguas", (1.00, 0.82, 0.20)),
    )
    target_pixels = labeling.target_counts.sum()
    region_colors = np.empty((labeling.region_ids.size, 3))
    for category, name, color in categories:
        selected = labeling.labels == category
        count = selected.sum()
        region_fraction = count / labeling.region_ids.size
        target_fraction = (
            f"{labeling.target_counts[selected].sum() / target_pixels:.2%}"
            if target_pixels else "indefinido (classe ausente)"
        )
        print(f"{name}: {count} regiões ({region_fraction:.2%}); "
              f"pixels da classe-alvo: {target_fraction}")
        region_colors[selected] = color

    figure, axes = comparison_panels([
        (rgb, "Imagem original"),
        (target_mask, "Ground truth da classe"),
        (mark_boundaries(rgb / 255.0, regions), "Segmentação SLIC"),
        (region_colors[labeling.pixel_indices], "Rotulagem supervisionada"),
    ])
    axes[3].legend(
        handles=[Patch(facecolor=color, label=name) for _, name, color in categories],
        loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3,
    )

    save_comparison(figure, Path(args.output))
    plt.show()


if __name__ == "__main__":
    main()
