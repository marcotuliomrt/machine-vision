import cv2
import matplotlib.pyplot as plt
import numpy as np

fig1_gray = cv2.imread("VM_Aula5_Figuras/Fig4A.png", cv2.IMREAD_GRAYSCALE) # fundo
fig2_gray = cv2.imread("VM_Aula5_Figuras/Fig4B.png", cv2.IMREAD_GRAYSCALE) # purturbacao


# def img_subtract(img1, img2, th):
#     (h, w) = img1.shape

#     fig1_gray32 = img1.astype(np.int32)
#     fig2_gray32 = img2.astype(np.int32)

#     fig_out = np.zeros((h,w), dtype = "uint8")


#     for i in range(h):
#         for j in range(w):
#             intens32 = abs(fig1_gray32[i,j] - fig2_gray32[i,j])
#             if intens32 > th:
#                 fig_out[i,j] = 255

#     return fig_out
        

# Optimized function with numpy functions
def img_subtract(img1, img2, th):
    (h, w) = img1.shape

    fig1_gray32 = img1.astype(np.int32)
    fig2_gray32 = img2.astype(np.int32)

    fig_out = np.zeros((h,w), dtype = "uint8")

    intens32 = abs(fig1_gray32 - fig2_gray32)
    fig_out = np.where(intens32 > th, 255, 0)
   
    return fig_out
        




plt.figure("")
plt.imshow(img_subtract(fig1_gray, fig2_gray, 5), cmap="gray")
plt.show()
