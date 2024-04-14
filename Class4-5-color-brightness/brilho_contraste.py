import cv2
import matplotlib.pyplot as plt
import numpy as np

fig_gray = cv2.imread("VM_Aula5_Figuras/Fig2.tif", cv2.IMREAD_GRAYSCALE)

(h, w) = fig_gray.shape

fig_gray32 = fig_gray.astype(np.int32)
fig_out = np.zeros((h,w), dtype = "uint8")
mult=3
soma=30


for i in range(h-1):
    for j in range(w-1):
    #faz a multiplicacao e soma (manipulando variaveis int32)
    #depois trunca o nro pra mantê-lo entre 0 e 255
    #vc ainda pode converter de volta par uint8
        intens32 = fig_gray32[i,j]*mult+soma
        fig_out[i,j] = np.clip( intens32, 0, 255).astype(np.uint8)


plt.figure("")
plt.imshow(fig_out, cmap="gray")
plt.show()
