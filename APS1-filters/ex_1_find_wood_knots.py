"""
Questão 1:
Dado o problema de criar um índice de qualidade de madeira, baseado na
proporção de nós e falhas mais escuras, considere a figura abaixo (Fig_APS1_1a).
Perceba que, na imagem coletada pela câmera, há sombras nas bordas da imagem
que não representam nós ou falha da madeira. Além disso, há o logotipo do
fabricante que também não representa falhas.
O objetivo é utilizar da soma ou subtração de imagens para remover essas
características não desejadas e posteriormente utilizar os programas já concluídos
para calcular a razão de área com nós e área sem nós.
Nota: A soma ou subtração de imagens é feita com uma máscara disponibilizada
(Fig_APS1_1b).
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread("Figuras_APS1/Fig_APS1_1a.bmp", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("Figuras_APS1/Fig_APS1_1b.bmp", cv2.IMREAD_GRAYSCALE)


def apply_mask(img, mask):
    (h, w) = img.shape

    fig_out = np.zeros((h,w), dtype = "uint8")

    for i in range(h):
        for j in range(w):
            if mask[i,j] == 255:
                fig_out[i,j] = 255
            else:
                fig_out[i,j] = img[i,j]

    return fig_out
          

def binarization(fig, threshold_inf, threshold_sup, after):
    (h, w) = fig.shape
    black = 0

    fig_bin = np.zeros((h, w), dtype = "uint8") # black image

    for i in range(h):
        for j in range(w):
            if ((fig[i, j] > threshold_inf) and (fig[i, j] < threshold_sup+1)):  # if the pixel is in the interval defines
                fig_bin[i, j] = after # make the pixel a defined value
            else:
                black+=1   

    percent = black*100/(h*w)
    return fig_bin, percent



masked = apply_mask(img,mask)

bla, percent = binarization(masked, 120, 255, 255)





print("Percentage of wood knots = ", percent)
plt.figure("figB")
plt.imshow(bla, cmap="gray")
plt.show()
