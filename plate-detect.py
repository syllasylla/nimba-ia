import cv2
import imutils
import numpy as np
import os
import time
from ultralytics import YOLO
import pytesseract
import tools.color_list as tls_color

import tools.plate_tools as plate_tools
import tools.showing as showing

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path = './images'
files_list = os.listdir(path)
nb_files = len(files_list)
pos = 0
key = None
model = YOLO('./models/nimba-ia.pt')
tab_color = [
    (10, 255, 255),  (255, 10, 255),  (255, 255, 10),  (10, 10, 255),  (10, 255, 10), (255, 10, 10),
    (255, 255, 255), (100, 255, 255), (255, 255, 100), (255, 100, 255) ]
waitSec = 100

def yolo_get_plate_xyxy_list(frame):
    dim_list = []
    yolo_res = model.predict(frame, verbose=False)
    for res in yolo_res:
        boxes = res.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            # x1, y1, x2, y2 = r
            dim_list.append(r)
            # w, h = x2 - x1, y2 - y1

            ''' if plate_tools.is_rectangle_plate(h, w):
                color = tab_color[0]
            else:
                color = tab_color[1] '''

            cv2.rectangle(img640x480, r[:2], r[2:], (255, 255, 10), 2)
            # cv2.putText(imgCopy, f'x1 x2 : ({x1} {x2}) y1 y2 : ({y1} {y2}) w : {w} h : {h}', (0,30 * cpt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tab_color[cpt], 1)
    if len(dim_list) < 1:
        return None
    return dim_list

def img_crop(plaque, x_ratio, y_ratio):
    h, w, __ = plaque.shape
    h0, w0 = int(h * y_ratio), int(w * x_ratio)
    h_pad = ((h - h0) // 2) - 1
    w_pad = ((w - w0) // 2) - 1
    img_center = plaque[h_pad : h - h_pad, w_pad : w - w_pad]
    return img_center

def print_plate_to_image(dim_list, image):
    if dim_list:
        H = 0
        i = 1
        for dim in dim_list:
            if plate_tools.is_bad_plate(dim):
                continue
            x1, y1, x2, y2 = dim
            h, w = y2 - y1, x2 - x1

            plaque = frame[y1:y2, x1:x2]
            img_center = img_crop(plaque, 0.8, 0.8)
            image[H:h + H, 0:w] = plaque

            #color = dominant_rgb(img_center)
            #tls_color.track_red(img_center, pytesseract, i)
            #tls_color.track_blue(img_center, pytesseract, i)
            tls_color.track_white(img_center, pytesseract, i)
            b, g, r = 200, 200, 10
            cv2.putText(imgCopy, f'color ({r},{g},{b}) ', (w, h + H), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 1)
            H = h + 50
            #cv2.imwrite(f'./plaque/{time.time()}.jpg', plaque)
            img_ocr = img_crop_corner(plaque, 0.25, 0.25)
            img_ocr = cv2.cvtColor(img_ocr, cv2.COLOR_BGR2GRAY)
            # str_ocr = pytesseract.image_to_string(img_ocr, lang='eng')
            # print(str_ocr)

def rotate10_and_print_to_img(dim_list, angle, image, padding=10, spacing=10):
    pad = padding // 2
    H = 0
    for dim in dim_list:
        x1, y1, x2, y2 = dim
        h, w = y2 - y1, x2 - x1
        plaque_init = frame[y1:y2, x1:x2]

        plaque = np.zeros((h+padding, w+padding, 3), dtype='uint8')
        plaque = plaque + 255
        plaque[pad:h+pad, pad:w+pad] = plaque_init

        img_rot = plaque
        img_rot = imutils.rotate(img_rot, angle)

        image[H:h + H + padding, 0:w + padding] = img_rot
        cv2.putText(image, f' w : {w} h : {h}', (w, h + H), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 10), 1)
        H = h + spacing

def rotate(image, angle):
    h, w, channel = image.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def expand_and_rotate_img(frame, dim, angle, padding=10):
    pad = padding // 2
    x1, y1, x2, y2 = dim
    h, w = y2 - y1, x2 - x1

    plaque = None
    try :
        plaque = frame[y1 - pad: y2 + pad, x1 - pad: x2 + pad]
        plaque = plaque.copy()
        plaque = rotate(plaque, angle)
    except:
        plaque = np.zeros((h + padding, w + padding, 3), dtype='uint8')
        plaque = plaque + 255
        plaque[pad: h + pad, pad: w + pad] = frame[y1:y2, x1:x2]
        plaque = plaque.copy()
        plaque = rotate(plaque, angle)

    return plaque

def img_crop_corner(img, yratio, xratio):
    img = showing.normalize_color(img)
    h, w, __ = img.shape
    pady = int(h * yratio)
    padx = int(w * xratio)
    img = img[pady:h-pady, padx:w-padx]
    return img.copy()



def dominant_rgb(img, pad=4):
    img &= 0b11111100
    #img = tls_color.k_means(img, 4)
    #img = showing.normalize_color(img)
    h, w, __ = img.shape
    pady = int(h * 0.25)
    padx = int(w * 0.25)
    img0 = img[pady:h-pady, padx:w-padx]
    a = img0.copy()

    cv2.imshow("dominap", img)

    colors, count = np.unique(a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    total = np.sum(count)
    dict_color = { }

    for i in range(len(count)):
        size = count[i]
        name_color = tls_color.color_name(colors[i])
        key = f"{size:010d}-{name_color}"
        dict_color[key] = colors[i]

    tls_color.show_color(dict_color)
    ind0 = count.argmax()
    count[ind0] = 0
    ind1 = count.argmax()
    count[ind1] = 0
    return colors[count.argmax()]

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)



t0 = time.time()
print(t0)
while True:
    fname = files_list[pos]
    filename = f'{path}/{fname}'
    img = cv2.imread(filename)
    img640x480 = imutils.resize(img, width=640, height=480)
    frame = img640x480.copy()
    imgCopy = img640x480.copy()

    imgCopy = imgCopy * 0
    imgRot10plus = imgCopy.copy()
    imgRot10minus = imgRot10plus.copy()

    dim_list = yolo_get_plate_xyxy_list(img640x480)
    if dim_list:
        print_plate_to_image(dim_list, imgCopy)

        for dim in dim_list:
            if plate_tools.is_bad_plate(dim):
                continue




    # imgVstack_1 = np.hstack([imgRot10plus, imgRot10minus])
    imgVstack_0 = np.hstack([img640x480, imgCopy])
    imgShow = imgVstack_0

    cv2.imshow('momo nimba', imgShow)
    key = cv2.waitKey(waitSec)
    if key == ord('q'): break
    elif key == ord('j'): # next
        pos = pos + 1
    elif key == ord('f'): # prev
        pos = pos - 1

    #pos = pos + 1
    if pos < 0: pos = 0
    elif pos >= nb_files: pos = nb_files - 1



cv2.destroyAllWindows()
print(time.time() - t0)
# img15_plus = expand_and_rotate_img(frame, dim, 30, padding=40)
# img15_minus = expand_and_rotate_img(frame, dim, -30, padding=40)
# cv2.imshow('img plus', img15_plus)
# cv2.imshow('img minus', img15_plus)
# rotate10_and_print_to_img(dim_list, 15, imgRot10plus, padding=40, spacing=80)
