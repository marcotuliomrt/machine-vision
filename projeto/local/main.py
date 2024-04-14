import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
# Initialize video capture
# cap = cv2.VideoCapture("./assets/Video2_Vedacao.mp4")
cap = cv2.VideoCapture("./assets/Video1_OK.avi")


# Set up the detector with default parameters.
params = cv2.SimpleBlobDetector_Params()
# Set blob color (0=black, 1=white)
params.filterByColor = False

# Importacao dp modelo treinado do Keras

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the trained model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()


# params.blobColor = 0

# Filter by Area
params.filterByArea = True
params.minArea = 400
params.maxArea = 200_000

# Filter by Circularity
params.filterByCircularity = False
# params.minCircularity = 0.2
# params.maxCircularity = 200

# Filter by Convexity
params.filterByConvexity = False
# params.minConvexity = 0.87
# params.maxConvexity = 1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.9
params.maxInertiaRatio = 1

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

img_exists = False
calculo_is_done = False
counter = 0

diametro, valor_ab = 0, 0
resultado_diametro, resultado_ab = False, False

df = pd.DataFrame(
    columns=["FILENAME", "TESTE - BORDA", "TESTE - SUPERFICIE",
             "DIAMETRO", "STATUS - DIAMETRO", "A/B", "STATUS - A/B"]
)


def calcula_relacao_eixo_a_e_b(ellipse):
    a, b = ellipse[1][0]/2, ellipse[1][1]/2
    teste_a_b = a/b

    return teste_a_b


def calcula_diametro(comprimento_px, largura_real_cm, comprimento_obj):

    # mede o comprimento da imagem em pixels
    comprimento_px = comprimento_px

    # calcula a relação de pixels por centímetro (ppc)
    ppc = comprimento_px / largura_real_cm

    # mede o comprimento de um objeto na imagem em pixels
    comprimento_objeto_px = comprimento_obj

    # converte o comprimento em pixels para centímetros
    comprimento_objeto_cm = comprimento_objeto_px / ppc

    raio = comprimento_objeto_cm/10

    return raio


def fun_diametro(img):
    contagem_pixels = []

    # Percorre cada linha da imagem
    for y in range(img.shape[0]):
        # Conta o número de pixels 255 na linha atual
        contagem = cv2.countNonZero(img[y])
        # Adiciona o número de pixels contados à lista
        contagem_pixels.append(contagem)
    # Encontra a linha com o maior número de pixels 255
    linha_mais_pixels = max(contagem_pixels)
    return linha_mais_pixels


TESTE_BORDA = "-"
TESTE_SUPERFICIE = "-"
STATUS_DIAMETRO = "-"
STATUS_AB = "-"
DIAMETRO = "-"
AB = "-"
FILENAME = ""

conta_frames = 0

while True:
    # Read frame from video
    ret, frame = cap.read()

    if not ret:
        break

    fig_borda = frame

    # Separa os canais HSV
    hsv = cv2.cvtColor(fig_borda, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    px = v[int(v.shape[0]/2), v.shape[1]-150]

    hsv_min = np.array([120, 0, 0])
    hsv_max = np.array([255, 150, 95])

    mask = cv2.inRange(fig_borda, hsv_min, hsv_max)

    result = cv2.bitwise_and(fig_borda, fig_borda, mask=~mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    _, fig_gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    blur = cv2.medianBlur(fig_gray, 5)

    kp = detector.detect(blur)

    if (img_exists and len(kp) == 0):
        counter += 1
        img_exists = False
        calculo_is_done = False

    if len(kp) > 0:
        img_exists = True

    if len(kp) > 0 and not calculo_is_done:
        if(conta_frames > 5):
            contours, _ = cv2.findContours(
                blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cnt = max(contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(cnt)

            _, radius = cv2.minEnclosingCircle(cnt)

            area_contour = cv2.contourArea(cnt)
            area_ellipse = np.pi * ellipse[1][0] * ellipse[1][1] / 4.0
            diff = area_ellipse - area_contour

            if(diff > 20):
                TESTE_BORDA = "Reprovado"
                data = {
                    "FILENAME": [FILENAME],
                    "TESTE - BORDA": [TESTE_BORDA],
                    "TESTE - SUPERFICIE": ["-"],
                    "DIAMETRO": ["-"],
                    "STATUS - DIAMETRO": ["-"],
                    "A/B": ["-"],
                    "STATUS - A/B": ["-"]

                }
                data = pd.DataFrame(data)
                df = pd.concat([df, data], ignore_index=True)
                continue
            else:
                TESTE_BORDA = "Aprovado"




            # teste de superficie

            # Resize the raw image into (224-height,224-width) pixels
            image = cv2.resize(fig_borda, (224, 224),
                               interpolation=cv2.INTER_AREA)

            # Make the image a numpy array and reshape it to the models input shape.
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

            # Normalize the image array
            image = (image / 127.5) - 1

            # Predicts the model
            prediction = model.predict(image)
            index = np.argmax(prediction)
            confidence_score = prediction[0][index]

            if(index == 1):
                TESTE_SUPERFICIE = "Reprovado"
                data = {
                    "FILENAME": [FILENAME],
                    "TESTE - BORDA": [TESTE_BORDA],
                    "TESTE - SUPERFICIE": [TESTE_SUPERFICIE],
                    "DIAMETRO": ["-"],
                    "STATUS - DIAMETRO": ["-"],
                    "A/B": ["-"],
                    "STATUS - A/B": ["-"]
                }
                data = pd.DataFrame(data)
                df = pd.concat([df, data], ignore_index=True)
                continue
            else:
                TESTE_SUPERFICIE = "Aprovado"



            linha_mais_pixels = fun_diametro(blur)
            diametro = calcula_diametro(
                fig_borda.shape[0], 756, linha_mais_pixels)

            if(diametro > 49.5 and diametro < 50.5):
                STATUS_DIAMETRO = "Aprovado"
                DIAMETRO = diametro
            else:
                STATUS_DIAMETRO = "Reprovado"
                DIAMETRO = diametro

            ab = calcula_relacao_eixo_a_e_b(ellipse)
            if(ab > 0.95 and ab < 1.05):
                AB = ab
                STATUS_AB = "Aprovado"
            else:
                AB = ab
                STATUS_AB = "Reprovado"

            data = {
                "FILENAME": [FILENAME],
                "TESTE - BORDA": [TESTE_BORDA],
                "TESTE - SUPERFICIE": [TESTE_SUPERFICIE],
                "DIAMETRO": [DIAMETRO],
                "STATUS - DIAMETRO": [STATUS_DIAMETRO],
                "A/B": [AB],
                "STATUS - A/B": [STATUS_AB],
            }
            data = pd.DataFrame(data)
            df = pd.concat([df, data], ignore_index=True)
            calculo_is_done = True
            conta_frames = 0
        else:
            conta_frames += 1

    img1_with_KPs = cv2.drawKeypoints(
        frame,
        kp,
        np.array([]),
        (0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if isinstance(DIAMETRO, float):
        DIAMETRO = round(DIAMETRO, 4)

    if isinstance(AB, float):
        AB = round(AB, 4)

    cv2.putText(img1_with_KPs, f'Contagem: {counter}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(img1_with_KPs, f'Diametro: {DIAMETRO} mm - Status: {STATUS_DIAMETRO}',
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(img1_with_KPs, f'Relacao A/B: {AB} - Status: {STATUS_AB}',
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(img1_with_KPs, f'Relacao A/B: {AB} - Status: {STATUS_AB}',
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    # Show frame with keypoints
    cv2.imshow('frame', img1_with_KPs)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
print(df)
