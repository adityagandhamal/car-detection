# Import required libraries
import cv2
import numpy as np

# Load the image
img = cv2.imread("./Data/test_img.jpg")

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade Classifier and detect the cars in the image
model = cv2.CascadeClassifier("haarcascade_car.xml")
cars = model.detectMultiScale(img_gray, 1.1, 2)

# Draw rectangles around the detected cars
for x, y, w, h in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show the image
cv2.imshow("image", img)

# On clicking Close, close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

