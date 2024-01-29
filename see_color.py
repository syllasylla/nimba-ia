import cv2
import numpy as np
import random
import tools.color_list as tls_color


colors = [
    [50, 50, 100], [50, 50, 150], [50, 50, 200], [50, 100, 150], [50, 100, 200], [100, 100, 150], [100, 100, 200], [100, 150, 200], [150, 150, 200]
]


img = np.zeros((1000, 100, 3), dtype=np.uint8)
i = 0
step = 30
j = 0
jMax = 00
for color in colors:
    j = j + 1
    if j < jMax: continue
    #if color in tls_color.bad_color:
    #    continue
    r, g, b = color[0], color[1], color[2]
    h0 = step * i
    cv2.rectangle(img, (0, h0), (1000, step + h0), (b, g, r), thickness=-1)
    txt = f'color ({r},{g},{b})'
    cv2.putText(img, txt, (0, h0 + step//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    i = i + 1


cv2.imshow("fenTitle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
