import numpy as np
import cv2

image = cv2.imread(r'data\train\benign\0000869.jpg')
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=30, minRadius=600, maxRadius=640)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)
        cv2.circle(output, (x, y), 3, (255, 0, 0), 3)

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.imshow("output", output)
cv2.waitKey(0)
