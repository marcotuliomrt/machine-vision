import cv2
import matplotlib.pyplot as plt
import numpy as np

fig_gray = cv2.imread("Figuras/Fig1b.bmp", cv2.IMREAD_GRAYSCALE)

if fig_gray is None:
    print("File not found")
    exit(0)


(h, w) = fig_gray.shape
black = 0

fig_bin = np.zeros((h, w), dtype = "uint8")

for i in range(h):
    for j in range(w):
        if fig_gray[i, j] > 135:
            fig_bin[i, j] = 255
        
        
# function that does the same thing, actualy its on a even lower level of abstraction
def binarization(fig, threshold_inf, threshold_sup, after):
    (h, w) = fig.shape
    black = 0

    fig_bin = np.zeros((h, w), dtype = "uint8") # black image

    for i in range(h):
        for j in range(w):
            if fig[i, j] < threshold_sup and fig[i, j] > threshold_inf:  # if the pixel is in the interval defines
                fig_bin[i, j] = after # make the pixel a defined value
  
    return fig_bin

img2 = binarization(fig_gray, 135, 255, 255)

plt.figure("fig_bw")
plt.imshow(fig_bin, cmap="gray")

plt.figure("img2")
plt.imshow(img2, cmap="gray")

plt.show()
