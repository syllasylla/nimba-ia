import cv2
import numpy as np


def color_5(color):
    if color < 51: return 50
    elif color < 102: return 100
    elif color < 153: return 150
    elif color < 204: return 200
    return 250

def normalize_color(plaque):
    # B G R
    h, w, channel = plaque.shape
    plate = plaque.copy()
    for y in range(h):
        for x in range(w):
            b, g, r = plate[y, x]
            b, g, r = color_5(b), color_5(g), color_5(r)
            plate[y, x] = [b, g, r]
    return plate



