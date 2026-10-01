import numpy as np
from .classes import DamageClass

def create_binary_mask(label: np.ndarray, damage_class: DamageClass) -> np.ndarray:
    """
    Cria uma máscara binária para a classe de dano especificada.

    Args:
        label (np.ndarray): A label original.
        damage_class (DamageClass): A classe de dano para a qual criar a máscara.

    Returns:
        np.ndarray: Máscara binária.
    """
    return (label == damage_class.value).astype(np.uint8)

