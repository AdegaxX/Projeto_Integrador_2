# Verificar as labels usando o roboflow: FUNCIONAL

import cv2
import supervision as sv
from inference import get_model

# 📷 Carrega a imagem
image = cv2.imread("homem2.png")  # Substitua pelo caminho correto da imagem

# 🤖 Carrega o modelo de detecção de rostos do Roboflow
model = get_model(model_id="detec_objetos/5", api_key="mp6z8hktgskSRsKdXT8Y")  # Insira sua API Key

# 🔍 Executa inferência na imagem
result = model.infer(image, confidence=0.3)

# 📦 Converte para objeto supervision.Detections
detections = sv.Detections.from_inference(result[0])

# 🖍️ Inicializa o BoxAnnotator
box_annotator = sv.BoxAnnotator()

# 🖼️ Anota a imagem
annotated_frame = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# 💾 Salva o resultado
cv2.imwrite("output_annotated.jpg", annotated_frame)
print("✅ Imagem anotada salva como 'output_annotated.jpg'")
