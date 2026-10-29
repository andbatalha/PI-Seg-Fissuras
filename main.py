from pathlib import Path
import time

import joblib

from dataset.index import build_dataset_index
from dataset.split import split_dataset, split_train_validation
from dataset.loader import load_image, load_label

from training import build_training_data, train_random_forest
from prediction import predict_image
from evaluation import calculate_evaluation_counts, calculate_metrics


def main():
    print("=== PI-SEG-FISSURAS ===")
    print("1 - Treinar modelo")
    print("2 - Validar modelo")
    print("3 - Predizer imagem")
    print("0 - Sair")

    opcao = input("Selecione uma opção: ")

    if opcao == "0":
        print("Encerrando...")
        return

    if opcao not in ("1", "2", "3"):
        print("Opção inválida")
        return

    # Configuração comum
    rgb_dir = Path("data/BFDD/RGB")
    labels_dir = Path("data/BFDD/Label")

    # Divisão do dataset
    if opcao in ("1", "2"):
        dataset_index = build_dataset_index(labels_dir)

        development_images, test_images = split_dataset(dataset_index)

        train_images, validation_images = split_train_validation(
            development_images,
            dataset_index
        )

    # OPÇÃO 1 - TREINAMENTO
    if opcao == "1":
        print("Treinamento selecionado")

        inicio = time.perf_counter()

        X_train, y_train = build_training_data(
            train_images,
            rgb_dir,
            labels_dir,
            max_background=2000,
            max_per_class=500
        )

        print(f"Preparação: {time.perf_counter() - inicio:.2f}s")

        inicio = time.perf_counter()

        model = train_random_forest(X_train, y_train)

        print(f"Treinamento: {time.perf_counter() - inicio:.2f}s")
        print("Classes:", model.classes_)

        Path("models").mkdir(exist_ok=True)

        joblib.dump(model, "models/random_forest_bg2000.joblib")

        print("Modelo salvo!")

    # OPÇÃO 2 - VALIDAÇÃO
    elif opcao == "2":
        print("Validação selecionada")

        model = joblib.load("models/random_forest_bg2000.joblib")

        total_counts = {
            classe: {"TP": 0, "FP": 0, "FN": 0}
            for classe in range(6)
        }

        inicio = time.perf_counter()

        # Percorrer as imagens de validação
        for index, image in enumerate(validation_images, start=1):
            image_path = rgb_dir / f"{image}.JPG"
            label_path = labels_dir / f"{image}.png"

            img = load_image(image_path)
            lbl = load_label(label_path)

            pred_mask = predict_image(model, img)

            calculated_counts = calculate_evaluation_counts(lbl, pred_mask)

            # Acumular TP, FP e FN
            for classe in range(6):
                total_counts[classe]["TP"] += calculated_counts[classe]["TP"]
                total_counts[classe]["FP"] += calculated_counts[classe]["FP"]
                total_counts[classe]["FN"] += calculated_counts[classe]["FN"]

            print(
                f"\rValidando imagens: {index}/{len(validation_images)}",
                end="",
                flush=True
            )

        # A partir daqui, estamos fora do for das imagens
        print()
        print(f"Validação: {time.perf_counter() - inicio:.2f}s")

        print("\nResultados globais da validação:")

        for classe, counts in total_counts.items():
            metrics = calculate_metrics(
                counts["TP"],
                counts["FP"],
                counts["FN"]
            )

            print(f"\nClasse {classe}:")
            print(f"  TP: {counts['TP']}")
            print(f"  FP: {counts['FP']}")
            print(f"  FN: {counts['FN']}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall:    {metrics['Recall']:.4f}")
            print(f"  IoU:       {metrics['IoU']:.4f}")
            print(f"  F1-score:  {metrics['F1']:.4f}")

    # OPÇÃO 3 - PREDIÇÃO
    elif opcao == "3":
        print("Predição selecionada")

if __name__ == "__main__":
    main()