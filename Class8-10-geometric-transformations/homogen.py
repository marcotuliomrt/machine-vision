
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/screw_u8.bmp", cv2.IMREAD_GRAYSCALE)

(h, w) = img_original.shape

img_original = img_original.astype("int32")

img_final = np.ones((h, w), dtype="int32")*255

scale = np.array([[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 1]])
trans = np.array([[1, 0, 200],
                  [0, 1, 100],
                  [0, 0,  1]])
alpha = np.pi/6
rot_trans = np.array([[np.cos(alpha), -np.sin(alpha), 200],
                      [np.sin(alpha),  np.cos(alpha), 100],
                      [     0,              0,         1]])                

for x in range(w):
    for y in range(h):
        before = np.array([x, y, 1])
        # after = np.array([u, w])
        u, v, _ = np.matmul(rot_trans, before)
        u = int(u)
        v = int(v)
        if u < w and v < h:
            img_final[v, u] = img_original[y, x]
        else:
            pass

# Display the pattern
plt.imshow(img_final.astype("uint8"), cmap="gray")
plt.show()