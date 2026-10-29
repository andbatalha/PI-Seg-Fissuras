import cv2
import numpy as np

from dataset.loader import load_label


def extract_features(image, gray):

    # 1. Canais RGB
    rgb_features = image.reshape(-1, 3)

    # 2. Intensidade em escala de cinza
    gray_features = gray.reshape(-1, 1)

    # 3. Média local (janela 3x3)
    media_local = cv2.blur(gray, (3, 3)).reshape(-1, 1)

    # Conversão para float32 para os cálculos seguintes
    gray_float = gray.astype(np.float32)

    # 4. Desvio-padrão local (janela 3x3)
    media = cv2.blur(gray_float, (3, 3))
    media_quadrados = cv2.blur(gray_float ** 2, (3, 3))

    variancia = np.maximum(media_quadrados - media ** 2, 0)
    desvio_padrao_local = np.sqrt(variancia).reshape(-1, 1)

    # 5. Magnitude do gradiente de Sobel (bordas)
    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.hypot(grad_x, grad_y).reshape(-1, 1)

    # 6. Combinação de todas as características em uma matriz
    features = np.column_stack((
        rgb_features,
        gray_features,
        media_local,
        desvio_padrao_local,
        magnitude
    )).astype(np.float32)

    return features

def sample_pixels(features, label, max_background, max_per_class, rng):
    """
    Amostra pixels de cada classe, com limite específico para o background.

    Args:
        features: matriz (327680, 7) com as características dos pixels.
        label: máscara (512, 640) com as classes reais.
        max_background: quantidade máxima de pixels da classe Background (0).
        max_per_class: quantidade máxima de pixels de cada uma das demais classes.
        rng: gerador de números aleatórios do NumPy.

    Returns:
        x_sample: características dos pixels selecionados.
        y_sample: classes correspondentes aos pixels selecionados.
    """
    y = label.reshape(-1)

    x_sample = []
    y_sample = []

    for classe in range(6):
        indices_classe = np.flatnonzero(y == classe)

        if classe == 0:
            quantidade = min(max_background, len(indices_classe))
        else:
            quantidade = min(max_per_class, len(indices_classe))

        indices_selecionados = rng.choice(
            indices_classe,
            size=quantidade,
            replace=False
        )

        x_sample.append(features[indices_selecionados])
        y_sample.append(y[indices_selecionados])

    x_sample = np.concatenate(x_sample, axis=0)
    y_sample = np.concatenate(y_sample, axis=0)

    return x_sample, y_sample


def count_dataset_pixels(image_names, labels_dir):
    total_counts = np.zeros(6, dtype=np.int64)

    for image_name in image_names:
        label_path = labels_dir / f"{image_name}.png"
        label = load_label(label_path)

        counts = np.bincount(label.reshape(-1), minlength=6)
        total_counts += counts

    return total_counts

