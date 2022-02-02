import cv2
import numpy as np

image = cv2.imread('resources/trafiko.jpg', cv2.IMREAD_UNCHANGED)
img_monochrome = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img_blurred = cv2.GaussianBlur(img_monochrome, (3,3), 0)

# Find edges
img_canny = cv2.Canny(img_blurred, 85, 255)
# Find contours
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image for contours
canvas = np.zeros(image.shape)
# draw the contours on the empty image
cv2.drawContours(canvas, contours, -1, (0,255,0), 3)

cv2.imshow("Object Contours", canvas)
cv2.imwrite('processed/outlines.jpg', canvas)

cv2.waitKey(5000)
cv2.destroyAllWindows()