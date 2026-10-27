import joblib

from pathlib import Path
import time
import numpy as np
import joblib


from dataset.index import build_dataset_index
from dataset.split import split_dataset, split_train_validation
from training import build_training_data, train_random_forest

rgb_dir = Path("data/BFDD/RGB")
labels_dir = Path("data/BFDD/Label")

# 1. Preparar divisão dos dados
dataset_index = build_dataset_index(labels_dir)

development_images, test_images = split_dataset(dataset_index)

train_images, validation_images = split_train_validation(
    development_images,
    dataset_index
)

print(f"Treinamento: {len(train_images)}")
print(f"Validação: {len(validation_images)}")
print(f"Teste: {len(test_images)}")

# 2. Construir amostras
inicio = time.perf_counter()

X_train, y_train = build_training_data(
    train_images,
    rgb_dir,
    labels_dir,
    max_per_class=500,
    seed=42
)

print("Shape X:", X_train.shape)
print("Shape y:", y_train.shape)
print("Distribuição:", np.bincount(y_train, minlength=6))

assert X_train.shape[0] == y_train.shape[0]
assert X_train.shape[1] == 7
assert np.isfinite(X_train).all()

print(f"Preparação: {time.perf_counter() - inicio:.2f}s")

# 3. Treinar
inicio = time.perf_counter()

model = train_random_forest(X_train, y_train)

print(f"Treinamento: {time.perf_counter() - inicio:.2f}s")
print("Classes:", model.classes_)

# 4. Salvar
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/random_forest.joblib")

print("Modelo salvo!")