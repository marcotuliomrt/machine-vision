import cv2
import matplotlib.pyplot as plt
import numpy as np

# fig_BGR = cv2.imread("Figuras/Fig2b.bmp")

fig_BGR = cv2.imread("lena_original.jpg")

fig_RGB =fig_BGR[:,:,::-1] 


if fig_BGR is None:
    print("File not found")
    exit(0)


def rgb_to_gray(color_img):
        grayImage = np.zeros(color_img.shape)
        R = np.array(color_img[:, :, 0])
        G = np.array(color_img[:, :, 1])
        B = np.array(color_img[:, :, 2])

        R = (R *0.299)
        G = (G *0.587)
        B = (B *0.114)

        Avg = (R+G+B)
        grayImage = color_img.copy()

        for i in range(3):
           grayImage[:,:,i] = Avg
           
        return grayImage       

# grayImage = np.zeros(fig_BGR.shape)
# R = np.array(fig_BGR[:, :, 1])


grayImage = rgb_to_gray(fig_BGR)  
plt.imshow(grayImage)
#plt.imshow(fig_RGB)
plt.show()

# plt.imshow(grayImage)
# plt.show()

