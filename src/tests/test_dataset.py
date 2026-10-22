from pathlib import Path
import numpy as np

from dataset.classes import DamageClass
from dataset.index import build_dataset_index
from dataset.split import split_dataset
from dataset.split import split_train_validation
from features import count_dataset_pixels
from training import build_training_data
from training import train_random_forest

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


train_images, validation_images = split_train_validation(
    development_images,
    dataset_index
)

print("Treinamento:", len(train_images))
print("Validação:", len(validation_images))
print("Teste:", len(test_images))
print("Total:", len(train_images) + len(validation_images) + len(test_images))

train_set = set(train_images)
validation_set = set(validation_images)
test_set = set(test_images)

print("Treino x Validação:", len(train_set & validation_set))
print("Treino x Teste:", len(train_set & test_set))
print("Validação x Teste:", len(validation_set & test_set))

assert len(train_images) == len(train_set)
assert len(validation_images) == len(validation_set)
assert len(test_images) == len(test_set)

train_counts = count_dataset_pixels(train_images, labels_dir)
validation_counts = count_dataset_pixels(validation_images, labels_dir)

print("Pixels por classe no treinamento:", train_counts)
print("Pixels por classe na validação:", validation_counts)


import time

inicio = time.perf_counter()
 
X_train, y_train = build_training_data(
    train_images,
    Path("data/BFDD/RGB"),
    Path("data/BFDD/Label"),
    max_per_class=500,
    seed=42
)

fim = time.perf_counter()

print(f"Tempo: {fim - inicio:.2f} segundos")
print("Shape X_train:", X_train.shape)
print("Shape y_train:", y_train.shape)
print("Distribuição:", np.bincount(y_train, minlength=6))

assert X_train.shape[0] == y_train.shape[0]
assert X_train.shape[1] == 7
assert np.isfinite(X_train).all()
assert np.all(np.bincount(y_train, minlength=6) > 0)

inicio = time.perf_counter()

model = train_random_forest(X_train, y_train)

fim = time.perf_counter()

print(f"Tempo de treinamento: {fim - inicio:.2f} segundos")
print("Quantidade de árvores:", len(model.estimators_))
print("Classes aprendidas:", model.classes_)