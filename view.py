import cv2
import os
import imutils
import ultralytics

import tls.tls_view as tls_v

model = ultralytics.YOLO('./models/nimba-ia.pt')

path_img = './images/'
list_dir = os.listdir(path_img)
size_list = len(list_dir)
i = 0
action = None

while True:
    fname = f'{path_img}/{list_dir[i]}'
    img = cv2.imread(fname)
    img640 = imutils.resize(img, width=640, height=480)

    yolo_res = tls_v.yolo_detection(model, img640)
    if yolo_res:
        for xyxy in yolo_res:
            r = xyxy
            cv2.rectangle(img640, r[:2], r[2:], (255, 255, 10), 2)
            # cv2.rectangle(img640, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), "")
    action = tls_v.show_img_getAction(img640)
    if tls_v.action_quit(action):
        break
    i = tls_v.action_to_indice(action, i, size_list)



cv2.destroyAllWindows()
