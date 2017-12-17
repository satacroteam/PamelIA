import cv2
import os

# Load an color image in grayscale
dir_input_focus = r"data/train/benign/"
# vec_list = os.listdir("D:\\challenge_isic\\app")
eye_cascade = cv2.CascadeClassifier('beauty_spot.xml')
dir_output_focus = "out_contouring"

# for j in vec_list:
img_path = "0000009.jpg"
img = cv2.imread(dir_input_focus + img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eye_cascade.detectMultiScale(img)
i = 1
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    eye = img[ey:ey + eh, ex:ex + ew]
    if len(img) > 60:
        cv2.imwrite(dir_output_focus + "eye_" + str(i) + "_" + img_path, eye)
        i = i + 1

# cv2.waitKey(0)
# cv2.destroyAllWindows()