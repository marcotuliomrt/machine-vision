

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/Fig0334a.tif", cv2.IMREAD_GRAYSCALE)
# Verify if the images have the same size


(h, w) = img_original.shape

n = 15
d = int((n-1)/2)
kernel = np.ones((n,n), dtype='int16')/n**2

img_final = np.zeros((h, w), dtype="uint8")

for i in range(d, h-d):
    for j in range(d, w-d):

        section = img_original[i-d:i+d+1, j-d:j+d+1]   # section of the original image that is gonna be altered with the kernel
        
        prod_img_kernel = kernel*section   # matrix convolution operation
        img_final[i, j] = prod_img_kernel.sum()



def binarization(fig, threshold_inf, threshold_sup, after):
    (h, w) = fig.shape
    black = 0

    fig_bin = np.zeros((h, w), dtype = "uint8") # black image

    for i in range(h):
        for j in range(w):
            if fig[i, j] < threshold_sup and fig[i, j] > threshold_inf:  # if the pixel is in the interval defines
                fig_bin[i, j] = after # make the pixel a defined value
  
    return fig_bin


threshold = np.amax(img_original)*0.25
img2 = binarization(img_final, threshold, 255, 255 )


plt.figure("img_final")
plt.imshow(img_final, cmap="gray")

plt.figure("img2")
plt.imshow(img2, cmap="gray")

plt.show()
