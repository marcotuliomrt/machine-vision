"""
Questão 4:
Dificilmente se obtém uma iluminação perfeitamente distribuída em toda a cena
e, por essa razão, partes da imagem mais distantes da fonte de luz ficam mais
escuras e as que se localizam mais próximas da fonte de luz ficam mais claras.
Para solucionar esse problema, é comum criar um padrão de iluminação a partir
da aquisição de uma imagem de uma superfície branca adquirida pela mesma
câmera e mesma iluminação. Esse padrão de luz é utilizado para se corrigir a
iluminação da cena original.
Assim, implementar um programa para corrigir a iluminação de uma imagem
médica como mostrada na figura (Fig_APS1_4a), utilizando o padrão de iluminação
(Fig_APS1_4b) e obtendo uma nova figura após a correção.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("Figuras_APS1/Fig_APS1_4a.bmp", cv2.IMREAD_GRAYSCALE)
light_profile = cv2.imread("Figuras_APS1/Fig_APS1_4b.bmp", cv2.IMREAD_GRAYSCALE)

# Verify if the images have the same size
if img.shape != light_profile.shape:
    raise ValueError("As imagens devem possuir o mesmo tamanho")


# Convert int8 to float32
img32 = img.astype(np.float32)
light_profile32 = light_profile.astype(np.float32)

# crete the correction matrix
k_matrix = np.mean(light_profile)/light_profile

# create the correted image
corrected = np.clip(img*k_matrix, 0, 255).astype(np.uint8)

plt.figure()
plt.imshow(corrected, cmap="gray")
plt.figure()
plt.imshow(light_profile, cmap="gray")
plt.figure()
plt.imshow(img, cmap="gray")
plt.show()
