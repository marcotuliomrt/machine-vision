# Turns the red parts into gray

import cv2
import matplotlib.pyplot as plt
import numpy as np
fig_BGR = cv2.imread("Figuras/Fig4.bmp")
fig_RGB = cv2.cvtColor(fig_BGR,cv2.COLOR_BGR2RGB)

if fig_RGB is None:
    print("File not found")
    exit(0)



(h, w, c) = fig_RGB.shape
fig_out = fig_RGB.copy()

for i in range(h):
    for j in range(w):
        if fig_out[i, j, 0] > 240 and fig_out[i, j, 1] == 0 and fig_out[i, j, 2] == 0:
            fig_out[i, j, 0] = 100
            fig_out[i, j, 1] = 100
            fig_out[i, j, 2] = 100

   



plt.figure("fig_out")
plt.imshow(fig_out, cmap="gray")
plt.show()
