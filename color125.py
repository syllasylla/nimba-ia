import cv2
import imutils
import numpy as np
import tools.color_list as tls_color

# img = cv2.imread("./images/IMG_20220310_073012.jpg")
# imgResized = imutils.resize(img, width=640, height=640)

imgNp = np.zeros((1000, 1900, 3), np.uint8) #/ dtype='uint8')
imgNp = imgNp + 255
colors = []


for r in range(0, 251, 50):
    for g in range(0, 251, 50):
        for b in range(0, 251, 50):
            colors.append([r, g, b])
i = 0
step = 50
line = 0
j = 0
jMax = 0 * 36
tabColor = [0, 6]
for color in colors:
    j = j + 1
    if j <= jMax: continue

    x = i * step
    x1 = x + step
    r, g, b = color
    cv2.rectangle(imgNp, (x, line), (x1, step), (b, g, r), thickness=-1)
    cv2.putText(imgNp, str(i), (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if i in tabColor:
        #tls_color.file_back(r, g, b)
        print(f'{i} {color}', end='\t')
    if i != 0 and i >= 36:
        break
    i = i + 1



cv2.imshow("fenTitle", imgNp)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("quit")
