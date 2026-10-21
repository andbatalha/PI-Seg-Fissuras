import numpy as np
import cv2

from dataset.loader import load_image

from pathlib import Path


def grayscale_luminosity(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image_gray



def clahe(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image)

    return image_clahe

image = Path("data/BFDD/RGB/DJI_20250624181809_0003.JPG")

img_rgb = load_image(image)

gray = grayscale_luminosity(img_rgb)

clahe_img = clahe(gray)

cv2.imwrite("clahe.png", clahe_img)



