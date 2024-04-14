# Goal: Control a robotic arm by the angle of each coordinate system relative to the one before

import cv2
import matplotlib.pyplot as plt
import numpy as np

arm_1 = cv2.imread("Figuras_APS2/Fig3_Arm1.bmp", cv2.IMREAD_GRAYSCALE)
arm_2 = cv2.imread("Figuras_APS2/Fig3_Arm2.bmp", cv2.IMREAD_GRAYSCALE)
base = cv2.imread("Figuras_APS2/Fig3_Base.bmp", cv2.IMREAD_GRAYSCALE)

arm_1 = arm_1.astype("int32")
arm_2 = arm_2.astype("int32")
base = base.astype("int32")


(h, w) = base.shape



alpha1 = np.pi/12
x1 = 140
y1 = 180
transf_1 = np.array([[np.cos(alpha1), -np.sin(alpha1), x1],
                      [np.sin(alpha1),  np.cos(alpha1), y1],
                      [     0,              0,         1]])  

alpha2 = np.pi/6
x2 = 285
y2 = 75
              
transf_2 = np.array([[np.cos(alpha2), -np.sin(alpha2), x2],
                      [np.sin(alpha2),  np.cos(alpha2), y2],
                      [     0,              0,         1]])                






# Uses the inverse of the transformation matrix to fill the final image
def fill_before(w, h, matrix, img): # Fill the pixel before
    img_final = np.zeros((h, w), dtype="int32")
    for u in range(w):
        for v in range(h):
            after = np.array([u, v, 1])
            # after = np.array([u, w])
            result = np.matmul(np.linalg.inv(matrix), after)
            x = int(result[0]/result[2])
            y = int(result[1]/result[2])
            #if x < w and y < h:
            if (x >= 0) and (x <= (w-1)) and (y >= 0) and (y <= (h-1)):
                img_final[v, u] = img[y, x]
            else:
                pass
    return img_final


positioned_arm_1 = fill_before(w, h, transf_1, arm_1)
positioned_arm_2 = fill_before(w, h, np.matmul(transf_1,transf_2), arm_2)
img_final = base + positioned_arm_1 + positioned_arm_2

# Display the pattern
plt.imshow(img_final.astype("uint8"), cmap="gray")
plt.show()