# Goal: find multiple apearances of a pattern -> use Normalized cross correlation (NCC)


import cv2
import matplotlib.pyplot as plt
import numpy as np
import Find_N_MinMaxValues

img_original = cv2.imread("Figuras_APS2/Fig2_Ferramentas_u8.bmp", cv2.IMREAD_GRAYSCALE)
img_original_RGB = cv2.imread("Figuras_APS2/Fig2_Ferramentas_u8.bmp")

pattern = cv2.imread("Figuras_APS2/Fig2_Padrao_u8.bmp", cv2.IMREAD_GRAYSCALE)



(h, w) = img_original.shape

(hp, wp) = pattern.shape

img_original = img_original.astype("float32")

img_final = np.zeros((h, w), dtype="float32")


# Function that calculates the Normalized Cross Correlation
def ncc(matrix1, matrix2):
    matrix1a = matrix1.astype('float32')*(1/255)
    matrix2a = matrix2.astype('float32')*(1/255)
    (h, w) = matrix1a.shape
    ncc = np.zeros((h, w), dtype="float32")
    a = (matrix1a*matrix2a).sum()
    b = (matrix1a**2).sum()
    c = (matrix2a**2).sum()
    ncc = a / np.sqrt(b*c)
    return ncc


# Loop that sweep the image getting the NCC between the pattern and the sections of the image
for i in range(h-hp):
    for j in range(w-wp):

        section = (img_original[i:i+hp, j:j+wp]).astype("float32")   # section of the original image that is gonna be altered with the pattern
        
        img_final[i, j] = ncc(section, pattern)   # matrix convolution operation
        



# Create the lists with x and y positions
x_list = []
y_list = []

for k in range(h - hp):
    for l in range(w -wp):

        if img_final[k, l] > 0.99:
            x_list.append(k)
            y_list.append(l)
         
print('First points list size = ', len(x_list))



# Filters the repeated posiitons (or too close)
i = 0
while i < (len(x_list)-1):
    if abs(x_list[i+1]-x_list[i]) < 5:
        x_list.pop(i)
        y_list.pop(i)
    else:
        i+=1
print('Filtered points list size = ', len(x_list))



# Draw a bounding box on each match found
for i in range(len(x_list)):
    x = int(x_list[i])
    y = int(y_list[i])
    cv2.rectangle(img_original_RGB, (y, x), (y + wp, x + hp), (0, 0, 255), 2)



#Display the pattern and the original image with bounding boxes
plt.imshow(img_original_RGB, cmap="gray")
plt.show()






