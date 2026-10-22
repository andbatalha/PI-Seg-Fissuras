import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from dataset.loader import load_image
from dataset.loader import load_label
from preprocessing import grayscale_luminosity
from features import sample_pixels
from features import extract_features

def build_training_data(train_images, rgb_dir, labels_dir, max_per_class=500, seed=42):
    rng = np.random.default_rng(seed)

    x_train_list = []
    y_train_list = []

    for image_name in train_images:
        image_path = rgb_dir / f"{image_name}.JPG"
        label_path = labels_dir / f"{image_name}.png"

        image = load_image(image_path)
        label = load_label(label_path)

        gray = grayscale_luminosity(image)

        features = extract_features(image, gray)

        x_sample, y_sample  = sample_pixels(features, label, max_per_class, rng)
        
        x_train_list.append(x_sample)
        y_train_list.append(y_sample)

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    return x_train, y_train


def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight= None,
            n_jobs=-1,
            random_state=42)

    model.fit(x_train, y_train)

    return model