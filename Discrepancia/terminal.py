import cv2
import numpy as np
import supervision as sv
from inference import get_model
from deepface import DeepFace

# Caminhos das imagens
IMG1_PATH = "homem.png"
IMG2_PATH = "homem2.png"

# Função para detectar uma "face composta" pelas partes
def detect_face(image_np, model, margin=40):
    result = model.infer(image_np, confidence=0.3)
    detections = sv.Detections.from_inference(result[0])

    # Inicializa listas para limites combinados
    x1s, y1s, x2s, y2s = [], [], [], []

    for bbox, label in zip(detections.xyxy, result[0].predictions):
        print(f"📦 Detecção: {label.class_name}")
        # Considera qualquer classe como parte do rosto
        x1, y1, x2, y2 = map(int, bbox)
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

    if x1s and y1s and x2s and y2s:
        # Combina os limites para formar um retângulo único
        x1 = max(0, min(x1s) - margin)
        y1 = max(0, min(y1s) - margin)
        x2 = min(image_np.shape[1], max(x2s) + margin)
        y2 = min(image_np.shape[0], max(y2s) + margin)

        return image_np[y1:y2, x1:x2]

    return None

# Carrega o modelo do Roboflow
model = get_model(model_id="detec_objetos/5", api_key="mp6z8hktgskSRsKdXT8Y")

# Carrega as imagens
img_rg = cv2.imread(IMG1_PATH)
img_nova = cv2.imread(IMG2_PATH)

# Detecta rostos com bounding boxes combinadas
face1 = detect_face(img_rg, model)
face2 = detect_face(img_nova, model)


if face1 is not None and face2 is not None:
    # Salva os rostos
    cv2.imwrite("temp1.jpg", face1)
    cv2.imwrite("temp2.jpg", face2)

    # E chama o verificador externo:
    import sys
    import subprocess

    subprocess.run([sys.executable, "deepTest.py"])


else:
    print("🚫 Não foi possível detectar a região facial nas imagens.")
