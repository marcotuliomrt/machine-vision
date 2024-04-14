
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/Fig0335a.tif", cv2.IMREAD_GRAYSCALE)


(h, w) = img_original.shape

n = 3
d = int((n-1)/2)
kernel = np.ones((n,n), dtype='int16')

img_final = np.zeros((h, w), dtype="uint8")

for i in range(d, h-d):
    for j in range(d, w-d):

        section = img_original[i-d:i+d+1, j-d:j+d+1]   # section of the original image that is gonna be altered with the kernel
        img_final[i, j] = np.median(section)



plt.figure("img_final")
plt.imshow(img_final, cmap="gray")

plt.show()
