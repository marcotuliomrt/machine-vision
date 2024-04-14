"""
Elabore um programa para abrir o “Video_APS1_3.avi” e processar cada frame
do arquivo a fim de destacar os carros (veja exemplos abaixo)
"""




import cv2
import matplotlib.pyplot as plt
import numpy as np
import time




def img_subtract(img1, img2, th):
    (h, w) = img1.shape

    fig1_gray32 = img1.astype(np.int32)
    fig2_gray32 = img2.astype(np.int32)

    fig_out = np.zeros((h,w), dtype = "uint8")

    intens32 = abs(fig1_gray32 - fig2_gray32)
    intens8 = intens32.astype(np.uint8)
    fig_out = np.where(intens8 > th, 255, 0).astype(np.uint8)
   
    return fig_out
        



cap = cv2.VideoCapture("Figuras_APS1/Video_APS1_3.avi")  # create the VIdeoCapture object from the webcam

fps = 60

_, first_frame = cap.read()  # get the frame
cv2.imwrite('background.jpg', first_frame)
background = cv2.imread("past_frame.jpg", cv2.IMREAD_GRAYSCALE)

while bool:
    bool, frame = cap.read()  # get the frame
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car = img_subtract(frame, background, 40)
    time.sleep(1/fps)
    #cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # displays the fps on the screen 
    
    cv2.imshow("Test video", car)
    # set the condition of manual brake by pressing "q" on the keyboard  
    if cv2.waitKey(1) & 0xFF==ord("k"):
        break

    


# release the webcam 
cap.release()
cv2.destroyAllWindows()




