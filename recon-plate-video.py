import cv2
import imutils
import os, time
from ultralytics import YOLO

model = YOLO('./models/neemba-tech-model.pt')
f_video = "./Video/video-5.mp4"
isPause = False

cap = cv2.VideoCapture(f_video)
while True:

    if isPause == False:
        try:
            __, frame = cap.read()
            frame = imutils.resize(frame, width=640, height=480)
        except:
            break

        results = model.predict(frame, verbose=False)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cv2.rectangle(frame, r[:2], r[2:], (255, 255, 10), 2)

        cv2.imshow("ande plate", frame)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        if key == ord(' '):
            isPause == True

    else: # isPause == True
        cv2.imshow('pause', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            isPause = False


cv2.destroyAllWindows()
