import cv2
import imutils
import os, time
from ultralytics import YOLO

model = YOLO('./models/neemba-tech-model.pt')
f_video = "./Video/video-8.mp4"
cap = cv2.VideoCapture(f_video)

while True:
    try:
        __, frame = cap.read()
        frame = imutils.resize(frame, width=640, height=480)
        img = frame.copy()
    except:
        break

    results = model.predict(frame, verbose=False)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cv2.rectangle(frame, r[:2], r[2:], (255, 255, 10), 2)
            fname = f'./images-plaques/{time.time()}.jpg'
            cv2.imwrite(fname, img)
            break
        break


    cv2.imshow("ande plate", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
