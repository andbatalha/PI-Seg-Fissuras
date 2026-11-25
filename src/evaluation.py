import numpy as np


def count_binary_pixels(reference: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    """Conta TP, FP, FN e TN em máscaras binárias 2D alinhadas."""
    if reference.ndim != 2 or reference.shape != prediction.shape or not reference.size:
        raise ValueError("Referência e predição devem ter as mesmas dimensões 2D não vazias.")

    real_positive = reference == 1
    predicted_positive = prediction == 1
    return {
        "TP": int(np.count_nonzero(real_positive & predicted_positive)),
        "FP": int(np.count_nonzero(~real_positive & predicted_positive)),
        "FN": int(np.count_nonzero(real_positive & ~predicted_positive)),
        "TN": int(np.count_nonzero((reference == 0) & (prediction == 0))),
    }


def count_region_pixels(stain: np.ndarray, non_stain: np.ndarray,
                        selected: np.ndarray) -> dict[str, int]:
    """Soma pixels de referência em regiões alinhadas às decisões binárias.

    As entradas são vetores por região e podem estar vazias. Cada região tem
    uma única decisão, aplicada tanto aos seus pixels Stain quanto aos demais.
    """
    if selected.ndim != 1 or selected.shape != stain.shape or selected.shape != non_stain.shape:
        raise ValueError("As contagens e as decisões devem ter a mesma forma unidimensional.")
    if selected.dtype != np.bool_:
        if not np.isin(selected, (0, 1)).all():
            raise ValueError("As decisões por região devem ser binárias.")
        selected = selected.astype(bool)
    return {
        "TP": int(stain[selected].sum()),
        "TN": int(non_stain[~selected].sum()),
        "FP": int(non_stain[selected].sum()),
        "FN": int(stain[~selected].sum()),
    }


def calculate_metrics(true_positive: int, false_positive: int,
                      false_negative: int) -> dict[str, float]:
    """Calcula métricas da classe positiva, usando zero se o denominador for nulo."""
    predicted_positive = true_positive + false_positive
    real_positive = true_positive + false_negative
    union = true_positive + false_negative + false_positive
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / real_positive if real_positive > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "Precision": precision,
        "Recall": recall,
        "IoU": true_positive / union if union > 0 else 0.0,
        "F1": f1_score,
    }


def pixel_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Calcula as métricas de Stain e a acurácia global das duas classes."""
    metrics = calculate_metrics(counts["TP"], counts["FP"], counts["FN"])
    total = sum(counts.values())
    metrics["Accuracy"] = (counts["TP"] + counts["TN"]) / total if total else 0.0
    return metrics
