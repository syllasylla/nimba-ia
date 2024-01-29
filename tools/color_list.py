import os
import re

import cv2
import numpy as np

def file_back(r, g, b):
    file = open("./res/back", "a")
    file.write(f"[{r},{g},{b}]\n")
    file.close()

tab_black = [
    [0, 0, 0],  [50, 50, 50]
]
tab_white = [
    [254, 254, 254], [200, 200, 200], [150, 150, 150],  [200, 200, 100], [200, 250, 100], [200, 250, 50], [250, 250, 250],
    [200, 200, 150],  [250, 200, 250], [200, 200, 250], [200, 250, 200], [150, 150, 100], [100, 100, 100], [150, 100, 100],
    [250, 150, 250], [200, 150, 100], [200, 250, 150], [150, 200, 150], [250, 250, 150], [250, 200, 200], [250, 200, 150],
    [250, 250, 200], [150, 200, 200], [200, 150, 150], [250, 150, 150], [200, 250, 250], [200, 150, 250], [150, 200, 250],
    [250, 200, 150], [250, 150, 200], [250, 150, 100], [250, 200, 100], [200, 150, 200]
]
tab_blue = [
    [0, 0, 250],  [50, 50, 250], [100, 0, 250], [100, 100, 250], [100, 100, 250], [0, 50, 200], [0, 0, 50], [100, 50, 250],
    [0, 50, 250], [0, 0, 100], [50, 50, 150], [50, 0, 250], [50, 100, 200], [50, 0, 200], [0, 0, 200], [50, 100, 250],
    [0, 0, 150], [50, 50, 200],  [0, 50, 150], [0, 100, 250], [50, 100, 150], [0, 200, 250], [150, 150, 200],
    [50, 200, 200], [50, 150, 200], [0, 150, 200], [100, 200, 250], [0, 50, 100], [0, 100, 200], [100, 150, 250],
    [50, 200, 250], [100, 150, 200], [0, 100, 150], [0, 200, 250], [50, 150, 250], [0, 100, 100], [0, 150, 150], [100, 250, 250], [0, 200, 200],
    [50, 150, 150], [50, 250, 250], [100, 200, 200], [0, 150, 250], [100, 50, 200], [100, 0, 200], [50, 0, 100], [100, 100, 150], [50, 0, 150],
    [100, 100, 200], [150, 100, 250], [50, 50, 100], [150, 150, 250]
]
tab_red = [
    [200, 0, 0], [250, 0, 50], [200, 0, 50], [250, 50, 250], [200, 0, 100], [250, 0, 100], [250, 50, 0], [200, 50, 50], [150, 50, 150],
    [250, 0, 200], [250, 0, 250], [150, 50, 100], [250, 50, 200], [50, 0, 0], [250, 0, 0], [100, 0, 0], [250, 50, 100], [150, 0, 0],
    [200, 50, 0], [250, 100, 50], [150, 50, 50], [200, 50, 0], [150, 0, 0], [250, 50, 50], [250, 100, 150],
    [250, 50, 150], [200, 50, 150], [200, 50, 100], [250, 100, 200], [250, 150, 150], [100, 0, 50], [100, 0, 100],
    [200, 50, 200], [200, 100, 150], [150, 0, 100], [200, 0, 150], [250, 0, 150],[150, 0, 150], [200, 0, 250], [200, 50, 250],
    [150, 0, 50], [200, 100, 200], [200, 100, 100], [100, 50, 50], [150, 0, 200],[100, 50, 100],
    [250, 100, 100], [200, 100, 250], [200, 0, 200], [250, 100, 250],
    [200, 100, 50], [150, 50, 0]
]
tab_green = [
    [0, 150, 50], [50, 50, 0], [100, 250, 100], [0, 200, 0], [50, 250, 0], [50, 150, 0], [100, 150, 0], [100, 200, 0], [50, 200, 50], [50, 150, 50],
    [100, 250, 50], [0, 250, 0], [100, 200, 100], [0, 250, 50], [0, 200, 50], [50, 200, 0], [100, 250, 0], [100, 200, 50], [50, 250, 50],
    [0, 100, 0], [50, 100, 50], [150, 250, 50], [100, 150, 100], [150, 200, 100], [0, 150, 0], [50, 100, 0], [0, 250, 150],
    [50, 200, 150], [100, 250, 200], [100, 200, 150], [0, 200, 100], [150, 250, 150],
    [50, 250, 150], [50, 150, 100], [0, 50, 0], [0, 100, 50], [100, 150, 50], [50, 200, 100], [50, 250, 100], [0, 150, 100], [0, 250, 100],
    [0, 200, 150], [100, 250, 150], [150, 250, 100], [150, 250, 0], [200, 250, 100], [0, 50, 50],
    [150, 250, 200], [50, 250, 200], [0, 250, 200], [150, 200, 50], [150, 150, 50],
    [0, 250, 250], [200, 250, 0], [150, 150, 0], [150, 200, 0], [100, 100, 50],
    [150, 250, 250], [0, 200, 50], [0, 100, 50], [100, 100, 0]
]
tab_yellow = [
    [250, 200, 50], [250, 200, 0], [200, 200, 100], [250, 250, 50], [200, 150, 0], [150, 100, 0],
    [200, 200, 0], [200, 150, 50], [250, 150, 0], [200, 200, 50], [250, 250, 100], [250, 100, 0],
    [250, 250, 0], [250, 150, 50], [200, 100, 0]
]

bad_color = [
    [50, 0, 50], [150, 100, 200], [150, 0, 250], [100, 50, 150], [150, 50, 250],
    [150, 50, 200], [150, 100, 150],  [50, 100, 100], [100, 0, 150],
    [100, 150, 150], [100, 50, 0], [150, 100, 50]
]

def color_name(bgr):
    rgb = [bgr[2], bgr[1], bgr[0]]

    for t in tab_black:
        if (t == rgb): return 'black'
    #if (tab_black == rgb).any() : return "black"
    '''if tab_white in rgb : return "white"
    if tab_red in rgb: return "red"
    if tab_green in rgb : return "green"
    if tab_blue in rgb : return "blue"
    if tab_yellow in rgb : return "yellow"'''
    return "black"

colors = [
    [50, 50, 100], [50, 50, 150], [50, 50, 200], [50, 100, 150], [50, 100, 200], [100, 100, 150], [100, 100, 200], [100, 150, 200], [150, 150, 200]
]


def show_color(dict_color):
    img = np.zeros((1000, 500, 3), dtype=np.uint8)
    i = 0
    step = 30
    j = 0
    jMax = 00
    dict_color = dict(sorted(dict_color.items(), key=lambda x : x[0], reverse=True))
    for key in dict_color:
        color = dict_color[key]
        name = key.split("-")[1]
        size = key.split("-")[0]
        size = int(size)

        j = j + 1
        if j < jMax: continue
        #if color in tls_color.bad_color:
        #    continue
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        h0 = step * i
        cv2.rectangle(img, (0, h0), (400, step + h0), (b, g, r), thickness=-1)
        txt = f'color ({r},{g},{b})'
        txt = f'{name} {size} {color}'
        cv2.putText(img, txt, (0, h0 + step//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        i = i + 1
        cv2.imshow('showcolor', img)


def k_means(img, K):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

#bad_color = bad_color + tab_red + tab_blue + tab_white + tab_black + tab_green + tab_yellow

def track_white(img, pytesseract, id):
    img_hsv =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tab_data = []
    step = 19
    for sensitivity in range(7, 255, 11):
        lower_white = np.array([0, 0, 255-sensitivity], np.uint8)
        upper_white = np.array([255, sensitivity, 255], np.uint8)
        mask = cv2.inRange(img_hsv, lower_white, upper_white)

        str_ocr = pytesseract.image_to_string(mask, lang='eng')
        str_ocr = re.sub(r'[^A-Z0-9]+', '', str_ocr)
        str = f'white-{sensitivity} : [{str_ocr}]'
        tab_data.append(str)
        show_text(tab_data, id)


def track_red(img, pytesseract, id):
    img_hsv =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tab_data = []

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    lower_mask = cv2.inRange(img_hsv, lower1, upper1)

    str_ocr = pytesseract.image_to_string(lower_mask, lang='eng')
    str_ocr = re.sub(r'[^A-Z0-9]+', '', str_ocr)
    str = f'red : [{str_ocr}]'
    tab_data.append(str)

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160, 100, 20])
    upper2 = np.array([179, 255, 255])
    upper_mask = cv2.inRange(img_hsv, lower2, upper2)

    str_ocr = pytesseract.image_to_string(upper_mask, lang='eng')
    str_ocr = re.sub(r'[^A-Z0-9]+', '', str_ocr)
    str = f'red : [{str_ocr}]'
    tab_data.append(str)
    full_mask = lower_mask + upper_mask
    str_ocr = pytesseract.image_to_string(full_mask, lang='eng')
    str_ocr = re.sub(r'[^A-Z0-9]+', '', str_ocr)
    str = f'red : [{str_ocr}]'
    tab_data.append(str)

    show_text(tab_data, id)


def track_blue(img, pytesseract, id):
    img_hsv =cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    tab_data = []

    blue_lower = np.array([100, 150, 0], np.uint8)
    blue_upper = np.array([140, 255, 255], np.uint8)
    mask = cv2.inRange(img_hsv, blue_lower, blue_upper)
    str_ocr = pytesseract.image_to_string(mask, lang='eng')
    str_ocr = re.sub(r'[^A-Z0-9]+', '', str_ocr)
    str = f'blue : [{str_ocr}]'
    tab_data.append(str)
    show_text(tab_data, id)

def show_text(tab_data, id):
    img = np.zeros((800, 400, 3), dtype=np.uint8)
    i = 0
    step = 30
    for val in tab_data:

        h0 = step * i
        cv2.putText(img, val, (0, h0 + step//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        i = i + 1
        cv2.imshow(f'img-to-txt-{id}', img)




