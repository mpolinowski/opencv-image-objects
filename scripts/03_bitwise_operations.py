import cv2
import numpy as np
from matplotlib import pyplot as plt

image_left = cv2.imread('resources/left.jpg')
image_right = cv2.imread('resources/right.jpg')

image_colour = cv2.imread('resources/on_fire.jpg', cv2.IMREAD_COLOR)

# Using bitwise_and operation on the given two images
merged_image = cv2.bitwise_and(image_left, image_right, mask = None)

# Displaying the merged image as the output on the screen
cv2.imshow('Left Image', image_left)
cv2.imshow('Right Image', image_right)
cv2.imshow('Merged Image', merged_image)

# Working with colour masks
rgb_conversion = cv2.cvtColor(image_colour, cv2.COLOR_BGR2RGB)

# Define colour range in RGB
dark_yellow = np.array([252, 170, 0])
bright_yellow = np.array([255, 205, 114])

# Select pixel within the defined colour range
mask_yellow = cv2.inRange(rgb_conversion, dark_yellow, bright_yellow)
colour_range = cv2.bitwise_and(image_colour, image_colour, mask=mask_yellow)

cv2.imshow('Original Image', image_colour)
cv2.imshow('Colour Range Selection', colour_range)

# Bitwise masking
image_colour_copy = image_colour.copy()
# Create mask with 100 rows, 300 columns and 3 colour channels
mask = np.zeros((100 , 300, 3))
# pos = (600, 600)
# set position of upper left corner and lower right corner of mask
var = image_colour_copy[200:(200+mask.shape[0]), 200:(200+mask.shape[1])] = mask
cv2.imshow('Masked Section', image_colour_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()