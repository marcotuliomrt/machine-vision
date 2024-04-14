"""
Elabore um programa para abrir o “Video_APS1_3.avi” e processar cada frame
do arquivo a fim de destacar os carros (veja exemplos abaixo)
"""




import cv2
import matplotlib.pyplot as plt
import numpy as np
import time



img = cv2.imread("Figuras_APS1/Fig_APS1_1a.bmp", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("Figuras_APS1/Fig_APS1_1b.bmp", cv2.IMREAD_GRAYSCALE)


def img_subtract(img1, img2, th):
    (h, w) = img1.shape

    fig1_gray32 = img1.astype(np.int32)
    fig2_gray32 = img2.astype(np.int32)

    fig_out = np.zeros((h,w), dtype = "uint8")


    for i in range(h):
        for j in range(w):
            intens32 = abs(fig1_gray32[i,j] - fig2_gray32[i,j])
            if intens32 > th:
                fig_out[i,j] = 255

    return fig_out
        

fig = img_subtract(img, mask, 5)




# obs: 0: integrates webcam, 2: usb camera
cap = cv2.VideoCapture("Figuras_APS1/Video_APS1_3.avi")  # create the VIdeoCapture object from the webcam

fps = 60

c_time = 0  # current time
p_time = 0  # previous time

bool, first_frame = cap.read()  # get the frame
cv2.imwrite('background.jpg', first_frame)
background = cv2.imread("past_frame.jpg", cv2.IMREAD_GRAYSCALE)

while True:
    bool, frame = cap.read()  # get the frame
    
    #past_frame = cv2.imread("past_frame.jpg", cv2.IMREAD_GRAYSCALE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car = img_subtract(frame, background, 15)

    time.sleep(1/fps)
    #cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # displays the fps on the screen 

    # set the condition of manual brake by pressing "q" on the keyboard  
    if cv2.waitKey(1) & 0xFF==ord("k"):
        break

    #cv2.imwrite('past_frame.jpg', frame)

    cv2.imshow("Test video", car)


# release the webcam 
cap.release()
cv2.destroyAllWindows()




