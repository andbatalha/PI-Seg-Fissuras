from pathlib import Path
import numpy as np

from dataset.loader import load_label
from dataset.masks import create_binary_mask
from dataset.classes import DamageClass
from dataset.index import get_present_classes


label_path = Path(
    "data/BFDD/Label/DJI_20250624181809_0003.png"
)
label = load_label(label_path)

classes = get_present_classes(label)

crack_mask = create_binary_mask(
    label,
    DamageClass.EROSION
)

print("Classes presentes na label:")
for damage_class in classes:
    print(f"- {damage_class.name} (valor: {damage_class.value})")

print("Label:", np.unique(label))

print("Erosion:")
print("Shape:", crack_mask.shape)
print("Valores:", np.unique(crack_mask))
print("Pixels:", np.sum(crack_mask))