import cv2
import imutils
import numpy as np
import os, time
from ultralytics import YOLO

import tools.showing as t_showing

def get_plaque(frame, r):
    x1, y1, x2, y2 = (r[0], r[1], r[2], r[3])
    plaque_0 = frame[y1:y2, x1:x2]
    plaque_1 = plaque_0.copy()
    h, w, channel = plaque_0.shape
    radius = int(h/5)
    colorCircle = (255, 255, 10)
    w2 = int(w/2)
    h2 = int(h/2)

    filename = f'./plaque/{time.time()}'
    cv2.circle(plaque_0, (w2, h2), radius, colorCircle, 1)
    fname = f'{filename}.0.jpg'
    cv2.imwrite(fname, plaque_0)

    plaque_1 = t_showing.normalize_color(plaque_1)
    fname = f'{filename}.1.jpg'
    cv2.circle(plaque_1, (w2, h2), radius, colorCircle, 1)
    cv2.imwrite(fname, plaque_1)



model = YOLO('./models/nimba-ia.pt')
count = 0

for fname in os.listdir('./images'):
    count = count + 1
    fname = f'./images/{fname}'
    img = cv2.imread(fname)
    img = imutils.resize(img, width=640, height=480)
    frame = img.copy()
    t0 = time.time()
    results = model.predict(img, verbose=False)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            get_plaque(frame, r)
            cv2.rectangle(img, r[:2], r[2:], (255, 255, 10), 2)

    cv2.imshow("ande plate", img)
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()