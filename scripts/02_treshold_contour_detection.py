import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('resources/trafiko.jpg', cv2.IMREAD_UNCHANGED)
img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set binary tresholds
ret, img_threshold = cv2.threshold(img_grayscale, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Object Contours", img_threshold)

# Find contours
contours, _ = cv2.findContours(img_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image for contours
canvas = np.zeros(image.shape)

i = 0

for contour in contours:
    if i == 0:
        i = 1
        continue

    # For each of the contours detected, the shape of the contours is
    # approximated using approxPolyDP() function and the
    # contours are drawn in the image using drawContours() function
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    # Draw polygons on the empty image
    cv2.drawContours(canvas, [contour], 0, (0, 255, 0), 5)
    # Find center of found shapes
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
    # Classifying shapes
    if len(approx) == 3:
       cv2.putText(canvas, 'Triangle', (x, y), cv2.QT_FONT_NORMAL, 0.6, (0, 255, 255), 2)
    elif len(approx) == 4:
       cv2.putText(canvas, 'Rectangle', (x, y), cv2.QT_FONT_NORMAL, 0.6, (255, 255, 255), 2)
    elif len(approx) == 6:
       cv2.putText(canvas, 'Hexagon', (x, y), cv2.QT_FONT_NORMAL, 0.6, (255, 0, 255), 2)
    elif 6 < len(approx) < 15:
       cv2.putText(canvas, 'Circle?', (x, y), cv2.QT_FONT_NORMAL, 0.6, (255, 255, 0), 2)

# Display results
cv2.imshow('Detected Shapes', canvas)
cv2.imwrite('processed/shapes.jpg', canvas)

cv2.waitKey(5000)
cv2.destroyAllWindows()