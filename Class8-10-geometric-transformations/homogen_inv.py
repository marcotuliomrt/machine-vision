import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/screw_u8.bmp", cv2.IMREAD_GRAYSCALE)

(h, w) = img_original.shape

img_original = img_original.astype("int32")



# scape
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

# Uses the inverse of the transformation matrix to fill the final image
def fill_before(w, h, matrix, img): # Fill the pixel before
    img_final = np.ones((h, w), dtype="int32")*255
    for u in range(w):
        for v in range(h):
            after = np.array([u, v, 1])
            # after = np.array([u, w])
            x, y, _ = np.matmul(np.linalg.inv(matrix), after)
            x = int(x)
            y = int(y)
            if x < w and y < h:
                img_final[v, u] = img[y, x]
            else:
                pass
    return img_final


img = fill_before(w, h, scale, img_original)

# Display the pattern
plt.imshow(img.astype("uint8"), cmap="gray")
plt.show()