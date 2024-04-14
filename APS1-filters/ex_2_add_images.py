"""
Questão 2:
Programar rotina de processamento digital de imagens para leitura dos arquivos
“Fig_APS1_2a” e “Fig_APS1_2b” e aplicação de função de junção das duas
imagens. Neste caso, deve-se unir a imagem da estrada com a imagem do disco
voador (sem o fundo branco), compondo assim a imagem final processada.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


background = cv2.imread("Figuras_APS1/Fig_APS1_2a.png", cv2.IMREAD_GRAYSCALE)
object = cv2.imread("Figuras_APS1/Fig_APS1_2b.png", cv2.IMREAD_GRAYSCALE)


def img_addition(img1, img2, th):
    (h, w) = img1.shape

    fig1_gray32 = img1.astype(np.int32)
    fig2_gray32 = img2.astype(np.int32)

    fig_out = np.zeros((h,w), dtype = "uint8")


    for i in range(h):
        for j in range(w):
            if fig2_gray32[i,j] != 255:
                fig_out[i,j] = fig2_gray32[i,j]
            else:
                fig_out[i,j] = fig1_gray32[i,j]
            
    return fig_out
        

added = img_addition(background, object, 5)



plt.figure("figA")
plt.imshow(background, cmap="gray")
plt.figure("figB")
plt.imshow(object, cmap="gray")
plt.figure("figC")
plt.imshow(added, cmap="gray")
plt.show()