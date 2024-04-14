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



(h, w) = img.shape


k_matrix= np.zeros((h,w), dtype = "float32") # correction factors' matrix (initialized with zeros)
sum = 0 # sum of the pixel values of the light_profile image
n = h*w # total number of pixels in the image


# Loop to sum the pixel values
for i in range(h):
    for j in range(w):
        sum+=light_profile[i,j]


const = sum/n # average intensity the light profile image pixels
print(const)

# Loop to make the correction factors' matrix
for i in range(h):
    for j in range(w):
        k_matrix[i,j] = const/light_profile[i,j]


float32 = img.astype(np.float32)
fig_out = np.zeros((h,w), dtype = "float32")

# Loop to apply the correction matrix on the original image
for i in range(h):
    for j in range(w):
        intens32 = float32[i,j]*k_matrix[i,j]
        fig_out[i,j] = np.clip(intens32, 0, 255).astype(np.float32)



plt.figure()
plt.imshow(fig_out, cmap="gray")
plt.figure()
plt.imshow(light_profile, cmap="gray")
plt.figure()
plt.imshow(img, cmap="gray")
plt.show()
