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