import cv2
import numpy as np

def show_img_getAction(img640):
    imgnew = cv2.cvtColor(img640, cv2.COLOR_BGR2HSV)
    while True:
        cv2.imshow('artp nimba', img640)
        cv2.imshow('artp nimba new', imgnew)
        key = cv2.waitKey(1)
        if key == ord('f'): action = 'back'
        elif key == ord('q'): action = 'quit'
        elif key == ord('j'): action = 'next'
        if key == ord('f') or key == ord('q') or key == ord('j'):
            return action


def action_quit(action):
    if action == 'quit':
        return True
    return False

def action_to_indice(action, i, size_list):
    if action == 'back':
        i = i - 1
    elif action == 'next':
        i = i + 1

    if i < 0:
        i = 0
    elif i >= size_list:
        i = size_list - 1
    return i


def yolo_detection(model, img640):
    list_dim = []
    yolo_res = model.predict(img640, verbose=False)
    for res in yolo_res:
        boxes = res.boxes.cpu().numpy()
        for box in boxes:
            dim = box.xyxy[0].astype(int)
            list_dim.append(dim)

    if len(list_dim) < 1:
        return False
    return list_dim

