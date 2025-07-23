import cv2
import numpy as np
import supervision as sv
from inference import get_model

# Função para desenhar detecções
def desenhar_deteccoes(image_np, model, confidence=0.7):
    result = model.infer(image_np, confidence=confidence)
    detections = sv.Detections.from_inference(result[0])

    for bbox, label in zip(detections.xyxy, result[0].predictions):
        x1, y1, x2, y2 = map(int, bbox)
        classe = label.class_name

        color = {
            "face": (0, 255, 0),
            "hat": (255, 255, 0),
            "glass": (255, 0, 255),
            "bar": (0, 165, 255),
            "mask": (255, 0, 0)
        }.get(classe.lower(), (200, 200, 200))

        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image_np,
            classe,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

    return image_np

# Carrega o modelo
print("Carregando modelo...")
model = get_model(model_id="detec_objetos/6", api_key="mp6z8hktgskSRsKdXT8Y")
print("Modelo carregado com sucesso.")

# Inicia webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processado = desenhar_deteccoes(frame.copy(), model)

    cv2.imshow("Deteccao", frame_processado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Encerrando...")
        break

cap.release()
cv2.destroyAllWindows()


# Desligar os alertas: pip install "inference[transformers,sam,clip,grounding-dino,yolo-world]"
# Testar outras cameras: cap = cv2.VideoCapture(1)  # ou 2, ou 3...
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Para Windows ou cv2.CAP_MSMF ou cv2.CAP_V4L2 (Linux).