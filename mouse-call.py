import cv2
import imutils
import os, time


winTitle = 'ande nimba'
frame = None

def mouse_event(event, x, y, flags, param):
    # cv2.EVENT_MOUSEMOVE
    # cv2.EVENT_LBUTTONDOWN
    # cv2.EVENT_RBUTTONDOWN
    # cv2.EVENT_MBUTTONDOWN
    if event == cv2.EVENT_MOUSEMOVE:
        tab = frame[y, x]
        print(f'pos ({x},{y})\t {tab} b: {tab[0]} g: {tab[1]} r: {tab[2]}')


img = cv2.imread('./images/IMG_20220310_073012.jpg')
img = imutils.resize(img, width=640, height=480)
frame = img
cv2.namedWindow(winTitle)
cv2.setMouseCallback(winTitle, mouse_event)

cv2.imshow(winTitle, img)
cv2.waitKey(-1)
cv2.destroyAllWindows()
