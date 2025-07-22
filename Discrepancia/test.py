import cv2
import mediapipe as mp
import urllib.request
import numpy as np
from deepface import DeepFace
import os

# Baixar imagens da internet
def baixar_imagem(url, caminho_saida):
    urllib.request.urlretrieve(url, caminho_saida)

# Recorta rosto com Mediapipe
def detectar_e_recortar_rosto(img_path, out_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        res = face_detection.process(img_rgb)

        if res.detections:
            for det in res.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = img.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                x, y = max(x, 0), max(y, 0)
                rosto = img[y:y+bh, x:x+bw]
                cv2.imwrite(out_path, rosto)
                return True
    return False

# URLs das imagens de teste
url1 = "https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img1.jpg"  # Obama
url2 = "https://raw.githubusercontent.com/serengil/deepface/master/tests/dataset/img2.jpg"  # Obama again

# Caminhos
img1_path = "obama1.jpg"
img2_path = "obama2.jpg"
crop1 = "obama1_crop.jpg"
crop2 = "obama2_crop.jpg"

# Baixar imagens
baixar_imagem(url1, img1_path)
baixar_imagem(url2, img2_path)

# Detectar e recortar
sucesso1 = detectar_e_recortar_rosto(img1_path, crop1)
sucesso2 = detectar_e_recortar_rosto(img2_path, crop2)

# Verificação com DeepFace
if sucesso1 and sucesso2:
    result = DeepFace.verify(img1_path=crop1, img2_path=crop2, enforce_detection=True)
    print("✅ Verificação bem-sucedida!")
    print("Mesma pessoa:", result["verified"])
    print("Distância:", result["distance"])
else:
    print("❌ Rosto não detectado em uma das imagens.")
