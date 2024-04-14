

import cv2
import matplotlib.pyplot as plt
import numpy as np

img_original = cv2.imread("Figuras_APS2/Fig4_Campo_Persp_u8.bmp", cv2.IMREAD_GRAYSCALE)

(h, w) = img_original.shape

img_original = img_original.astype("int32")


# Get the positions-> x,y:present image (8 DOF transformed)   u,v: rectified image
x0 = 102
y0 = 12.5
u0 = 0
v0 = 0 

x1 = 570
y1 = 22
u1 = 400
v1 = 0

x2 = 636
y2 = 458
u2 = 400
v2 = 500

x3 = 6
y3 = 220
u3 = 0
v3 = 500



# ---------- Projective transformation matrix: 8 DOF ---------------------

A = np.array([[x0, y0, 1, 0, 0, 0, -x0*u0, -y0*u0],
              [x1, y1, 1, 0, 0, 0, -x1*u1, -y1*u1],
              [x2, y2, 1, 0, 0, 0, -x2*u2, -y2*u2],
              [x3, y3, 1, 0, 0, 0, -x3*u3, -y3*u3],
              [0, 0, 0, x0, y0, 1, -x0*v0, -y0*v0],
              [0, 0, 0, x1, y1, 1, -x1*v1, -y1*v1],
              [0, 0, 0, x2, y2, 1, -x2*v2, -y2*v2],
              [0, 0, 0, x3, y3, 1, -x3*v3, -y3*v3]])

blob = np.array([[u0],
                 [u1],
                 [u2],
                 [u3],
                 [v0],
                 [v1],
                 [v2],
                 [v3]])

var = np.matmul(np.linalg.inv(A),blob)

H = np.array([[var[0][0], var[1][0], var[2][0]],
              [var[3][0], var[4][0], var[5][0]],
              [var[6][0], var[7][0],    1  ]], dtype="float32")





# ------------- Apply the transformation -------------------
                     
# Uses the inverse of the transformation matrix to fill the final image
def fill_before(w, h, matrix, img): # Fill the pixel before
    img_final = np.ones((h, w), dtype="int32")*255
    for u in range(w):
        for v in range(h):
            after = np.array([u, v, 1])
            # after = np.array([u, w])
            result = np.matmul(np.linalg.inv(matrix), after)
            x = int(result[0]/result[2])
            y = int(result[1]/result[2])
            #if x < w and y < h:
            if (x >= 0) and (x <= (w-1)) and (y >= 0) and (y <= (h-1)):
                img_final[v, u] = img[y, x]
            else:
                pass
    return img_final


gray = fill_before(w, h, H, img_original)

# Code to save the rectified image to work on it in other file
# cv2.imwrite("./img_transformed.jpg", img_transformed)







# --------------- Find the impediments -------------



def binarization(fig, threshold_inf, threshold_sup, after):
    (h, w) = fig.shape
   
    fig_bin = np.zeros((h, w), dtype = "uint8")

    for i in range(h):
        for j in range(w):
            if ((fig[i, j] >= threshold_inf) and (fig[i, j] < threshold_sup+1)):  # if the pixel is in the interval defines
                fig_bin[i, j] = after # make the pixel a defined value
   
    return fig_bin


# Apply a threshold in the image to get it binarized 
white_players = binarization(gray, 250, 255, 255)
black_players = binarization(gray, 0, 10, 255)
players = white_players + black_players # backgraund in back and the players in white




# Identify the x values of the left side of the players -> the edges of the left side -> list: values
values = []
for i in np.arange(1, h, 1): # sweeps in y
    for j in np.arange(1, w-350, 1): # sweeps in x
        if players[i, j-1] == 0 and players[i, j] == 255: #and j < min_white_x:
            values.append(j)

# print("values: ", values)




# Get the indexes of the list values where there is an singularity-> where the x values of one player change to the values of another one 
section_indexes = []
diferences = []
k = 1
while k < len(values):
    diferences.append(values[k] - values[k-1])
    if abs(values[k] - values[k-1]) > 100:
        section_indexes.append(k-1)
    k+=1


#plt.plot(diferences) # grath to identify where the x values of one player finishes and start the values of the other -> peaks on the graph

# Add an start and a finish indexes to the list (the original one has just the intermidiate values)
section_indexes.insert(0, 0)
section_indexes.append(len(values))
# print("section_indexes: ", section_indexes)


# Get the minimum values of the intervals defined on the section_indexes -> each interval represent the first half of the x values of the player
min_values = []
k = 0
while k < len(section_indexes)-1:
    min_values.append(min(values[section_indexes[k]+5 : section_indexes[k+1]-5]))
    k+=1

# print("min_values: ", min_values)

# Because the image sweep happens from top to bottom, the order of the values on the list min_values can tell which player is 

print("The order is from top to bottom")
for h in range(len(min_values)):
    print(" Minimum x position of player ", h+1, " is ", min_values[h])

print("Há impedimento pois o terceiro jogador (de cima pra baixo) esta na frente do primeiro")


#plt.imshow(gray.astype("uint8"), cmap="gray")




fig = plt.figure()

# show original image
fig.add_subplot(121)
plt.title(' image 1')
plt.set_cmap('gray')
plt.imshow(gray)

fig.add_subplot(122)
plt.title('Players pixels diferences (sections)')
plt.plot(np.arange(0, len(diferences), 1), diferences)

plt.show() 


