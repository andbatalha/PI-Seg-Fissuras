from pathlib import Path
import joblib
import numpy as np
from numpy.char import index

from dataset.index import build_dataset_index
from dataset.split import split_dataset, split_train_validation
from dataset.loader import load_image, load_label
from prediction import predict_image
from evaluation import calculate_evaluation_counts

rgb_dir = Path("data/BFDD/RGB")
labels_dir = Path("data/BFDD/Label")

dataset_index = build_dataset_index(labels_dir)

development_images, test_images = split_dataset(dataset_index)

train_images, validation_images = split_train_validation(development_images, dataset_index)

model = joblib.load("models/random_forest.joblib")

total_counts = {
    classe: {"TP": 0, "FP": 0, "FN": 0}
    for classe in range(6)
}

for index, image in enumerate(validation_images, start=1):
    image_path = rgb_dir/f"{image}.JPG"
    label_path = labels_dir/f"{image}.png"

    img = load_image(image_path)
    lbl = load_label(label_path)

    pred_mask = predict_image(model, img)

    calculated_counts = calculate_evaluation_counts(lbl, pred_mask)

    for classe in range(6):
        total_counts[classe]["TP"] += calculated_counts[classe]["TP"]
        total_counts[classe]["FP"] += calculated_counts[classe]["FP"]
        total_counts[classe]["FN"] += calculated_counts[classe]["FN"]

    print(f"Processando imagem {index}/{len(validation_images)}")


print("\nContagens totais:")
for classe, counts in total_counts.items():
    print(f"Classe {classe}: {counts}")