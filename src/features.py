import numpy as np
from pathlib import Path

from dataset.loader import load_image
from preprocessing import grayscale_luminosity


def extract_features(image, gray):

    rgb_features = image.reshape(-1, 3)

    gray_features = gray.reshape(-1, 1)

    features = np.float32(np.column_stack((rgb_features, gray_features)))

    return features


image = Path("data/BFDD/RGB/DJI_20250624181809_0003.JPG")

img_rgb = load_image(image)

gray = grayscale_luminosity(img_rgb)

features = extract_features(img_rgb, gray)

print("Shape RGB:", img_rgb.shape)
print("Shape Gray:", gray.shape)
print("Shape Features:", features.shape)
print("Dtype:", features.dtype)