import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/Brick_U8.bmp", cv2.IMREAD_GRAYSCALE)

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

perspec = np.array([[   1,   0,  0],
                    [ 0.12,  1,  0],
                    [0.001,  0,  1]])
                     
# Uses the inverse of the transformation matrix to fill the final image
def fill_before(w, h, matrix): # Fill the pixel before
    img_final = np.ones((h, w), dtype="int32")*255
    for u in range(w):
        for v in range(h):
            after = np.array([u, v, 1])
            # after = np.array([u, w])
            result = np.matmul(np.linalg.inv(matrix), after)
            x = int(result[0]/result[2])
            y = int(result[1]/result[2])
            #if x < w and y < h:
            if (x >= 0) and (x <= (w-1)) and (y >= 0) and (y <= (h-1)):
            #if (x <= (w-1)) and (y <= (h-1)):
                img_final[v, u] = img_original[y, x]
            else:
                pass
    return img_final


img = fill_before(w, h, perspec)

# Display the pattern
plt.imshow(img.astype("uint8"), cmap="gray")
plt.show()