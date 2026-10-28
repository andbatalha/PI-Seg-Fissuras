import cv2
import numpy as np
from pathlib import Path

from dataset.index import build_dataset_index
from dataset.loader import load_image
from dataset.loader import load_label
from preprocessing import grayscale_luminosity
from dataset.split import split_dataset
from features import extract_features, count_dataset_pixels, sample_pixels

image = Path("data/BFDD/RGB/DJI_20250624185000_0100.JPG")

img_rgb = load_image(image)

gray = grayscale_luminosity(img_rgb)

features = extract_features(img_rgb, gray)

print("Shape RGB:", img_rgb.shape)
print("Shape Gray:", gray.shape)
print("Shape Features:", features.shape)
print("Dtype:", features.dtype)

labels_dir = Path("data/BFDD/Label")

dataset_index = build_dataset_index(labels_dir)

development_images, test_images = split_dataset(dataset_index)


total_counts = count_dataset_pixels(development_images, labels_dir)

print("Pixels por classe:", total_counts)
print("Total:", total_counts.sum())


features = extract_features(img_rgb, gray)

print("Shape:", features.shape)
print("Dtype:", features.dtype)
print("Valores inválidos:", np.isnan(features).sum())
print("Desvio-padrão mínimo:", features[:, 5].min())
print("Desvio-padrão máximo:", features[:, 5].max())

label = load_label(Path("data/BFDD/Label/DJI_20250624185000_0100.png"))

rng = np.random.default_rng(42)

x_sample, y_sample = sample_pixels(features, label, 500, rng)

print("Shape X:", x_sample.shape)
print("Shape y:", y_sample.shape)
print("Distribuição:", np.bincount(y_sample, minlength=6))