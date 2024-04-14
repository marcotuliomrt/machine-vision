
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras/Fig0342a.tif", cv2.IMREAD_GRAYSCALE)


(h, w) = img_original.shape

n = 3
d = int((n-1)/2)
kernel_x = np.array([[-1, 0, 1], 
                      [-1, 0, 1],
                      [-1, 0, 1]],dtype='float32')
kernel_y = np.array([[-1,-1,-1], 
                      [ 0, 0, 0],
                      [ 1, 1, 1]],dtype='float32')

img_final_x = np.zeros((h, w), dtype="float32")
img_final_y = np.zeros((h, w), dtype="float32")


for i in range(d, h-d):
    for j in range(d, w-d):

        section = img_original[i-d:i+d+1, j-d:j+d+1]   # section of the original image that is gonna be altered with the kernel

        
        prod_img_kernel_x = kernel_x*section.astype("float32")   # matrix convolution operation
        img_final_x[i, j] = abs(prod_img_kernel_x.sum())
        
        prod_img_kernel_y = kernel_y*section.astype("float32")   # matrix convolution operation
        img_final_y[i, j] = abs(prod_img_kernel_y.sum())

        

print(kernel_x)

img_final_xy = np.clip(img_final_x + img_final_y, 0, 255).astype("uint8")

plt.figure("img_final_x")
plt.imshow(img_final_x.astype("uint8"), cmap="gray")
plt.figure("img_final_y")
plt.imshow(img_final_y.astype("uint8"), cmap="gray")
plt.figure("img_final")
plt.imshow(img_final_xy, cmap="gray")
plt.show()
