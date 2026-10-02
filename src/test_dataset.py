from pathlib import Path
import numpy as np
from numpy.ma import count

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



combination_groups = {}

for image_name, classes in dataset_index.items():
    damage_classes = []

    for damage_class in classes:
        if damage_class != DamageClass.BACKGROUND:
            damage_classes.append(damage_class)

    combination = tuple(damage_classes)

    if combination not in combination_groups:
        combination_groups[combination] = []

    combination_groups[combination].append(image_name)

test_images = []
development_images = []
   
for combination, image_names in combination_groups.items():

    names = []

    for damage_class in combination:
        names.append(damage_class.name)

    if len(combination) == 0:
        combination_name = "SEM DANOS"
    else:
        combination_name = " + ".join(names)

    print(f"{combination_name}: {len(image_names)} imagens")

    image_names = sorted(image_names)
    test_count = round(len(image_names) * 0.2)

    if test_count == 0:
        development_images.extend(image_names)
    else:
        test_images.extend(image_names[-test_count:])
        development_images.extend(image_names[:-test_count])


    print(f"Quantidade de imagens para teste: {test_count}")

print("\n=== RESULTADO DO SPLIT ===")
print(f"Teste: {len(test_images)} imagens")
print(f"Desenvolvimento: {len(development_images)} imagens")
print(f"Total: {len(test_images) + len(development_images)} imagens")

test_class_count = {}
for image_name in test_images:
    classes = dataset_index[image_name]

    for damage_class in classes:
        if damage_class == DamageClass.BACKGROUND:
            continue
        if damage_class not in test_class_count:
            test_class_count[damage_class] = 0

        test_class_count[damage_class] += 1


overlap = set(test_images) & set(development_images)

print(f"Overlap: {len(overlap)}")


print("\n=== CONTAGEM DE CLASSES PARA TESTE ===")
for damage_class, count in test_class_count.items():
    print(f"{damage_class.name}: {count}")
