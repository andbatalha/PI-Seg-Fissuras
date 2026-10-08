from pathlib import Path

from dataset.classes import DamageClass
from dataset.index import build_dataset_index
from dataset.split import split_dataset

labels_dir = Path("data/BFDD/Label")

dataset_index = build_dataset_index(labels_dir)

development_images, test_images = split_dataset(dataset_index)

print("Quantidade de imagens:", len(dataset_index))


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
