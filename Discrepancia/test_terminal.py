import cv2
import numpy as np
import supervision as sv
from inference import get_model
from deepface import DeepFace

# Caminhos das imagens (modifique conforme seus arquivos)
IMG1_PATH = "homem.png"     # foto do RG
IMG2_PATH = "homem2.png"    # foto atual

# Função para detectar o rosto com base no label 'face'
def detect_face(image_np, model):
    result = model.infer(image_np, confidence=0.3)
    detections = sv.Detections.from_inference(result[0])

    faces = [
        (bbox, label)
        for bbox, label in zip(detections.xyxy, result[0].predictions)
        if label["class"].lower() == "face"
    ]

    if len(faces) > 0:
        x1, y1, x2, y2 = map(int, faces[0][0])
        return image_np[y1:y2, x1:x2]
    return None

# Carrega o modelo do Roboflow
model = get_model(model_id="detec_objetos/5", api_key="mp6z8hktgskSRsKdXT8Y")

# Carrega as imagens
img_rg = cv2.imread(IMG1_PATH)
img_nova = cv2.imread(IMG2_PATH)

# Detecta rostos nas imagens
face1 = detect_face(img_rg, model)
face2 = detect_face(img_nova, model)

if face1 is not None and face2 is not None:
    # Salva os rostos temporariamente
    cv2.imwrite("temp1.jpg", face1)
    cv2.imwrite("temp2.jpg", face2)

    # Compara os rostos com DeepFace
    print("🔍 Comparando rostos com DeepFace...")
    try:
        result = DeepFace.verify(img1_path="temp1.jpg", img2_path="temp2.jpg", enforce_detection=False)
        distance = result["distance"]
        print(f"📏 Distância facial: {distance:.4f}")
        if distance > 0.6:
            print("⚠️ A pessoa está diferente. Pode ser necessário refazer o RG.")
        else:
            print("✅ As imagens são semelhantes. Não parece necessário refazer o RG.")
    except Exception as e:
        print(f"Erro ao comparar rostos: {e}")
else:
    print("❌ Não foi possível detectar o rosto em uma das imagens.")
