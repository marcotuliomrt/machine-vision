

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/Onde_Esta_Panda1_U8.bmp", cv2.IMREAD_GRAYSCALE)
img_original_RGB = cv2.imread("Figuras/Onde_Esta_Panda1_U8.bmp")

pattern = cv2.imread("Figuras/Panda_Padrao_u8.bmp", cv2.IMREAD_GRAYSCALE)

(h, w) = img_original.shape

(hp, wp) = pattern.shape

img_original = img_original.astype("int32")
pattern2 = pattern.astype("int32")

img_final = np.ones((h, w), dtype="int32")


for i in range(h-hp):
    for j in range(w-wp):

        section = (img_original[i:i+hp, j:j+wp]).astype("int32")   # section of the original image that is gonna be altered with the pattern
        
        prod_img_pattern = (abs(pattern2 - section)).sum()   # matrix convolution operation
        img_final[i, j] = prod_img_pattern



min = 255*hp*wp*2
x_min = 0
y_min = 0
for k in range(h - hp):
    for l in range(w -wp):

        if img_final[k, l] < min:
            min = img_final[k, l]
            x_min = k
            y_min = l

print(min)
print(x_min, y_min)

x = cv2.rectangle(img_original_RGB, (y_min, x_min), (y_min+wp, x_min+hp), (0, 0, 255), 2)

# Display the pattern
plt.imshow(x)
plt.show()


