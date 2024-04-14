import cv2
import matplotlib.pyplot as plt
import numpy as np



fig_gray1= cv2.imread("VM_Aula5_Figuras/Fig0316a.tif", cv2.IMREAD_GRAYSCALE)
fig_gray2= cv2.imread("VM_Aula5_Figuras/Fig0316b.tif", cv2.IMREAD_GRAYSCALE)
fig_gray3= cv2.imread("VM_Aula5_Figuras/Fig0316c.tif", cv2.IMREAD_GRAYSCALE)
fig_gray4= cv2.imread("VM_Aula5_Figuras/Fig0316d.tif", cv2.IMREAD_GRAYSCALE)

channels = [0]
mask = None
bins = [256]
ranges = [0, 256]
histogram1 = cv2.calcHist(fig_gray1, channels, mask, bins, ranges)
histogram2 = cv2.calcHist(fig_gray2, channels, mask, bins, ranges)
histogram3 = cv2.calcHist(fig_gray3, channels, mask, bins, ranges)
histogram4 = cv2.calcHist(fig_gray4, channels, mask, bins, ranges)



# plt.figure("Hist")
plt.plot(histogram2, color='k')
plt.show()


  
# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 2
columns = 2

  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
#plt.imshow(fig_gray1)

plt.axis('off')
plt.title("First")
plt.plot(histogram1)
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing histogram
#plt.imshow(fig_gray2)

plt.axis('off')
plt.title("Second")
plt.plot(histogram2)
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing fig_gray
#plt.imshow(fig_gray3)

plt.axis('off')
plt.title("Third")
plt.plot(histogram3)
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  
# showing fig_gray
#plt.imshow(fig_gray4)

plt.axis('off')
plt.title("Fourth")
plt.plot(histogram4)
plt.show()









# fig = cv2.imread("Figuras/Fig5.bmp", cv2.IMREAD_GRAYSCALE)

# if fig is None:
#     print("File not found")
#     exit(0)


# (h, w) = fig.shape
# color_list = []

# fig_bin = np.zeros((h, w), dtype = "uint8")

# for i in range(h):
#     for j in range(w):
#         if fig[i, j] > 100:
#             fig_bin[i, j] = 255
#         else:
#             black+=1
        
# knots = black*100/(h*w)

# print("Percentage of wood knots = ", knots)
# plt.figure("figB")
# plt.imshow(fig_bin, cmap="gray")
# plt.show()
