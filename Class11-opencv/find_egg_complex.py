
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_gray = cv2.imread("Figuras/Ovos_u8.bmp", cv2.IMREAD_GRAYSCALE)
ret, img1_bin = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)


kernel = np.ones((8, 8), np.uint8)
img1_bin = cv2.erode(img1_bin, kernel, iterations=3)




# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
# Set blob color (0=black, 255=white)
params.filterByColor = False
params.blobColor = 0
# Filter by Area
params.filterByArea = True
params.minArea = 10
params.maxArea = 500
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
# #params.maxCircularity = 1.2
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.7
#params.maxConvexity = 1
# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
#params.maxInertiaRatio = 1
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs
KP = detector.detect(img1_bin)
print("Nro de blobs: ",len(KP))
# List parameters (X,Y,size,ang) of each detected keypoints
img1_text = cv2.cvtColor(img1_bin,cv2.COLOR_GRAY2RGB)
i=1
for KPi in KP:
    print("Blob_", i, ": X= ", KPi.pt[0], " Y= ", KPi.pt[1], " size=", KPi.size**2, " ang=", KPi.angle)
    img1_text = cv2.putText(img1_text, str(i), (int(KPi.pt[0]),int(KPi.pt[1])), cv2.FONT_HERSHEY_PLAIN, 1,
    (0,0,255))
    i=i+1




# Draw detected blobs as red circles.
img1_with_KPs = cv2.drawKeypoints(img1_text, KP, np.array([]), (0,0,255),
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#display image (with blobs)
cv2.imshow("Img1 with keypoints", img1_with_KPs)
#aguarda uma tecla
cv2.waitKey(0) 