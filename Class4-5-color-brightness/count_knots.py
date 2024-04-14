import cv2
import matplotlib.pyplot as plt
import numpy as np


fig = cv2.imread("Figuras/Fig5.bmp", cv2.IMREAD_GRAYSCALE)

if fig is None:
    print("File not found")
    exit(0)

plt.figure("fig")
plt.imshow(fig)

(h, w) = fig.shape
black = 0

fig_bin = np.zeros((h, w), dtype = "uint8")

for i in range(h):
    for j in range(w):
        if fig[i, j] > 100:
            fig_bin[i, j] = 255
        else:
            black+=1
        
knots = black*100/(h*w)

print("Percentage of wood knots = ", knots)
plt.figure("figB")
plt.imshow(fig_bin, cmap="gray")
plt.show()
