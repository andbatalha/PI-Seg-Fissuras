from pathlib import Path
import joblib
import numpy as np

from dataset.index import build_dataset_index
from dataset.split import split_dataset, split_train_validation
from dataset.loader import load_image, load_label
from prediction import predict_image

rgb_dir = Path("data/BFDD/RGB")
labels_dir = Path("data/BFDD/Label")

dataset_index = build_dataset_index(labels_dir)

development_images, test_images = split_dataset(dataset_index)

train_images, validation_images = split_train_validation(development_images, dataset_index)

model = joblib.load("models/random_forest.joblib")

image_name = validation_images[0]

image = load_image(rgb_dir/f"{image_name}.JPG")
label = load_label(labels_dir/f"{image_name}.png")

predicted_mask = predict_image(model, image)

print("Imagem:", image_name)
print("Shape da imagem:", image.shape)
print("Shape da label:", label.shape)
print("Shape da previsão:", predicted_mask.shape)

print("Classes reais:", np.unique(label))
print("Classes previstas:", np.unique(predicted_mask))

real_counts = np.bincount(label.reshape(-1))
predicted_counts = np.bincount(predicted_mask.reshape(-1))

print("Pixels reais:", real_counts)
print("Pixels previstos:", predicted_counts)