import numpy as np
import cv2

from dataset.loader import load_image

from pathlib import Path


def grayscale_luminosity(image):

    altura, largura = image.shape[:2]

    image_gray = np.zeros((altura, largura), dtype=np.uint8)

    for y in range(altura):
        for x in range(largura):
            R = image[y][x][0]
            G = image[y][x][1]
            B = image[y][x][2]

            tom_cinza = (0.299 * R) + (0.587 * G) + (0.114 * B)

            image_gray[y][x] = round(tom_cinza)

    return image_gray

image = Path("data/BFDD/RGB/DJI_20250624181809_0003.JPG")

img_rgb = load_image(image)



