import cv2
import numpy as np

img_normal = cv2.imread(r'data\train\benign\0000869.jpg', 1)
img = cv2.imread(r'data\train\benign\0005537.jpg', 0)
img = cv2.equalizeHist(img)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=30, minRadius=600, maxRadius=615)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img_normal, (i[0], i[1]), i[2], (0, 255, 0), 2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img_normal)
cv2.waitKey(0)
cv2.destroyAllWindows()
