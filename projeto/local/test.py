import cv2
import numpy as np

# Load image
img = cv2.imread('image.jpg')

# Get image dimensions
height, width = img.shape[:2]

# Set rotation angle in degrees
angle = 45

# Calculate rotation matrix
M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

# Apply rotation to image
rotated_img = cv2.warpAffine(img, M, (width, height))

# Display the original and rotated images
cv2.imshow('Original', img)
cv2.imshow('Rotated', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()