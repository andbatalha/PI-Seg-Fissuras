import numpy as np

from dataset.loader import load_image
from dataset.loader import load_label

def build_training_data(train_images, rgb_dir, labels_dir, max_per_class=500, seed=42):
    rng = np.random.default_rng(seed)

    x_train_list = []
    y_train_list = []

    for image_name in train_images:
        image_path = rgb_dir / f"{image_name}.JPG"
        label_path = labels_dir / f"{image_name}.png"

        image = load_image(image_path)
        label = load_label(label_path)