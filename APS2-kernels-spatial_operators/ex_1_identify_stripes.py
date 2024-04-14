# Goal: identify the angle of stripes on an image
# 1. Filter the fabric threads to leave the contrast just on the stripes -> apply blur
# 2. Apply a threshold to get a bin image
# 3. Extract the information of edge -> derivative kernel
# 4. Get the information if there are more edges in x, y or between or even get the angle
# 5. Map the edges information into readable labels


import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras_APS2/Fig1_Tecido3.bmp", cv2.IMREAD_GRAYSCALE)


#---------------- 1 -------------------------

(h, w) = img_original.shape

n = 9
d = int((n-1)/2)
kernel = np.ones((n,n), dtype="float64")
sigma = 8

# Kernel construction loop
for i in range(-d, d+1):
    for j in range(-d, d+1):
        kernel[i+d, j+d] = (1/(2*np.pi*sigma**2))*np.exp(-(i**2 + j**2)/(2*sigma**2))

# Normalize the kernel
kernel /= np.sum(kernel)
# x=np.max(kernel)
# kernel /= x


img_blur = np.zeros((h, w), dtype="uint8")

for i in range(d, h-d):
    for j in range(d, w-d):

        section = img_original[i-d:i+d+1, j-d:j+d+1]   # section of the original image that is gonna be altered with the kernel
        
        prod_img_kernel = kernel*section   # matrix convolution operation
        img_blur[i, j] = prod_img_kernel.sum()


#---------------- 2 ---------------------------

for i in range(h):
    for j in range(w):
        if img_blur[i, j] > 150:
            img_blur[i, j] = 255
        else:
            img_blur[i, j] = 0



# --------------- 3 ----------------------------

n2 = 3
d2 = int((n2-1)/2)
kernel_x = np.array([[-1, 0, 1], 
                      [-1, 0, 1],
                      [-1, 0, 1]],dtype='float32')
kernel_y = np.array([[-1,-1,-1], 
                      [ 0, 0, 0],
                      [ 1, 1, 1]],dtype='float32')

img_sobel_x = np.zeros((h, w), dtype="float32")
img_sobel_y = np.zeros((h, w), dtype="float32")


for i in range(d2, h-d2):
    for j in range(d2, w-d2):

        section = img_blur[i-d2:i+d2+1, j-d2:j+d2+1]   # section of the original image that is gonna be altered with the kernel

        prod_img_kernel_x = kernel_x*section.astype("float32")   # matrix convolution operation
        img_sobel_x[i, j] = abs(prod_img_kernel_x.sum())
        
        prod_img_kernel_y = kernel_y*section.astype("float32")   # matrix convolution operation
        img_sobel_y[i, j] = abs(prod_img_kernel_y.sum())


# ---------------- 4 ---------------------------

white_x = 0
for i in range(h):
    for j in range(w):
        if img_sobel_x[i, j] > 150:
            white_x+=1
     
white_y = 0
for i in range(h):
    for j in range(w):
        if img_sobel_y[i, j] > 150:
            white_y+=1
     
ratio_x = white_x/(white_x+white_y)
ratio_y = white_y/(white_x+white_y)


print("ratio_x", ratio_x)
print("ratio_y", ratio_y)

# --------------- 5 ----------------------------


if ratio_x > 0.7 and ratio_y < 0.3:
    print("Vertical stripes")
if ratio_x < 0.3 and ratio_y > 0.7:
    print("Horizontal stripes")
if ratio_x > 0.3 and ratio_x < 0.7:
    print("Diagonal stripes")


 
#img_sobel_xy = np.clip(img_sobel_x + img_sobel_y, 0, 255).astype("uint8")

# plt.figure("img_sobel_x")
# plt.imshow(img_sobel_x.astype("uint8"), cmap="gray")
# plt.figure("img_sobel_y")
# plt.imshow(img_sobel_y.astype("uint8"), cmap="gray")
# plt.figure("img_sobel")
# plt.imshow(img_sobel_xy, cmap="gray")
# plt.show()
