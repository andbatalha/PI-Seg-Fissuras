from pathlib import Path
import numpy as np

from dataset.loader import load_label
from dataset.masks import create_binary_mask
from dataset.classes import DamageClass
from dataset.index import get_present_classes
from dataset.index import build_dataset_index


label_path = Path(
    "data/BFDD/Label/DJI_20250624181809_0003.png"
)
label = load_label(label_path)

classes = get_present_classes(label)

crack_mask = create_binary_mask(
    label,
    DamageClass.EROSION
)

labels_dir = Path("data/BFDD/Label")

dataset_index = build_dataset_index(labels_dir)

print("Quantidade de imagens:", len(dataset_index))



combination_count = {}

for classes in dataset_index.values():
    damage_classes = []

    for damage_class in classes:
        if damage_class != DamageClass.BACKGROUND:
            damage_classes.append(damage_class)

    combination = tuple(damage_classes)

    if combination not in combination_count:
        combination_count[combination] = 1
    else:
        combination_count[combination] += 1

for combination, count in combination_count.items():
    names = []

    for damage_class in combination:
        names.append(damage_class.name)

    combination_name = " + ".join(names)

    print(f"{combination_name}: {count} imagens")

