import numpy as np
from pathlib import Path

from dataset.index import build_dataset_index
from dataset.loader import load_image
from dataset.loader import load_label
from preprocessing import grayscale_luminosity
from dataset.split import split_dataset


def extract_features(image, gray):

    rgb_features = image.reshape(-1, 3)

    gray_features = gray.reshape(-1, 1)

    features = np.float32(np.column_stack((rgb_features, gray_features)))

    return features


image = Path("data/BFDD/RGB/DJI_20250624185000_0100.JPG")



img_rgb = load_image(image)

gray = grayscale_luminosity(img_rgb)

features = extract_features(img_rgb, gray)

print("Shape RGB:", img_rgb.shape)
print("Shape Gray:", gray.shape)
print("Shape Features:", features.shape)
print("Dtype:", features.dtype)

""" label_path = Path("data/BFDD/Label/DJI_20250624185000_0100.png")

label = load_label(label_path)

classes, contagem = np.unique(label, return_counts=True)

total_pixels = label.size

for classe, quantidade in zip(classes, contagem):
    porcentagem = (quantidade / total_pixels) * 100

    print(f"Classe: {classe}, Quantidade: {quantidade}, Porcentagem: {porcentagem:.2f}%") """


def count_dataset_pixels(image_names, labels_dir):
    total_counts = np.zeros(6, dtype=np.int64)

    for image_name in image_names:
        label_path = labels_dir / f"{image_name}.png"
        label = load_label(label_path)

        counts = np.bincount(label.reshape(-1), minlength=6)
        total_counts += counts

    return total_counts

labels_dir = Path("data/BFDD/Label")

dataset_index = build_dataset_index(labels_dir)

development_images, test_images = split_dataset(dataset_index)

labels_dir = Path("data/BFDD/Label")

total_counts = count_dataset_pixels(development_images, labels_dir)

print("Pixels por classe:", total_counts)
print("Total:", total_counts.sum())