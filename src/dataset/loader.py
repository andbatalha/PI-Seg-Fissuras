from pathlib import Path
import numpy as np
import cv2


def load_image(image_path):
    """
    Carrega uma imagem a partir do caminho fornecido.

    Args:
        image_path (Path): Caminho para a imagem.

    Returns:
        np.ndarray: Imagem carregada.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte de BGR para RGB  
    
    return image_rgb



def load_label(label_path):
    """
    Carrega uma label a partir do caminho fornecido.

    Args:
        label_path (Path): Caminho para a label.

    Returns:
        np.ndarray: Label carregada.
    """
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

    if label is None:
        raise FileNotFoundError(f"Não foi possível abrir a label: {label_path}")

    return label
