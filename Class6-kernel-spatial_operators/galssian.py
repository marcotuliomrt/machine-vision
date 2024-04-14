

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/Fig1b.bmp", cv2.IMREAD_GRAYSCALE)

(h, w) = img_original.shape

n = 9
d = int((n-1)/2)
kernel = np.ones((n,n), dtype="float64")
sigma = 8

# Kernel construction loop
for i in range(-d, d+1):
    for j in range(-d, d+1):
        kernel[i+d, j+d] = (1/(2*np.pi*sigma**2))*np.exp(-(i**2 + j**2)/(2*sigma**2))

# Normalize the kernel
kernel /= np.sum(kernel)
# x=np.max(kernel)
# kernel /= x


img_gaussian_blur = np.zeros((h, w), dtype="uint8")

for i in range(d, h-d):
    for j in range(d, w-d):

        section = img_original[i-d:i+d+1, j-d:j+d+1]   # section of the original image that is gonna be altered with the kernel
        
        prod_img_kernel = kernel*section   # matrix convolution operation
        img_gaussian_blur[i, j] = prod_img_kernel.sum()



print(kernel)

# Display the kernel
plt.imshow(img_gaussian_blur, cmap="gray")
plt.show()