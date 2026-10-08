import numpy as np

from .classes import DamageClass
from .index import get_present_classes
from .index import build_dataset_index

def split_dataset(dataset_index):
    combination_groups = {}

    for image_name, classes in dataset_index.items():
        damage_classes = []

        for damage_class in classes:
            if damage_class != DamageClass.BACKGROUND:
                damage_classes.append(damage_class)

        combination = tuple(damage_classes)

        if combination not in combination_groups:
            combination_groups[combination] = []

        combination_groups[combination].append(image_name)

    development_images = []
    test_images = []

    for combination, image_names in combination_groups.items():
        names = []

        image_names = sorted(image_names)
        test_count = round(len(image_names) * 0.2)

        if test_count == 0:
            development_images.extend(image_names)
        else:
            test_images.extend(image_names[-test_count:])
            development_images.extend(image_names[:-test_count])

    return development_images, test_images

