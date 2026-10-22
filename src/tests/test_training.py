from pathlib import Path

from dataset.split import split_train_validation
from dataset.split import split_dataset
from dataset.index import build_dataset_index

labels_dir = P

dataset_index = build_dataset_index(labels_dir)

development_images, test_images = split_dataset(dataset_index)

train_images, validation_images = split_train_validation(
    development_images,
    dataset_index
)

print("Treinamento:", len(train_images))
print("Validação:", len(validation_images))
print("Teste:", len(test_images))
print("Total:", len(train_images) + len(validation_images) + len(test_images))