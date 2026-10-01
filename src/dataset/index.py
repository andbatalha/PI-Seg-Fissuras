from pathlib import Path

import numpy as np

from .loader import load_label
from .classes import DamageClass

def get_present_classes(label):
    
    values = np.unique(label)
    classes = []

    for value in values:
        
        damage_class = DamageClass(value)
        classes.append(damage_class)
    
    return classes


def build_dataset_index(labels_dir: Path):
    label_paths = list(labels_dir.glob("*.png"))

    dataset_index = {}

    for label_path in label_paths:
        label = load_label(label_path)
        classes = get_present_classes(label)

        dataset_index[label_path.stem] = classes

    return dataset_index    
