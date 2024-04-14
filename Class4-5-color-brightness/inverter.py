import cv2
import matplotlib.pyplot as plt
import numpy as np

fig = cv2.imread("VM_Aula5_Figuras/Fig0304a.tif", cv2.IMREAD_GRAYSCALE)

if fig is None:
    print("File not found")
    exit(0)


fig2 = 255 - fig

plt.figure("fig")
plt.imshow(fig2, cmap="gray")
plt.show()