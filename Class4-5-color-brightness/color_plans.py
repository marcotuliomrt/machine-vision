import cv2
import numpy as np
import matplotlib.pyplot as plt
fig = cv2.imread("Figuras/Fig4.bmp")
[B,G,R] = cv2.split(fig)

# Takes apart the image channels
plt.figure("RED")
plt.imshow(R, cmap='gray')
plt.figure("GREEN")
plt.imshow(G, cmap='gray')
plt.figure("BLUE")
plt.imshow(B, cmap='gray')

# Merge the chanels together
(h, w) = B.shape

fig_bin = np.zeros((h, w), dtype = "uint8")

fig2 = cv2.merge((R, G, fig_bin))
plt.imshow(fig2)
plt.show()

