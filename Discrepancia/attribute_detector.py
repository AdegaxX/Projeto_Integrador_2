import cv2
from ultralytics import YOLO

# Carrega modelo custom para acessórios faciais
# model = YOLO('acessorios.pt')  # Usar esse se eu quiser treinar o modelo para detectar outras coisas

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Download do modelo pré-treinado (modelo simples) precisa de treino
model = YOLO('yolov8n.pt')


def detect_attributes(img_path):
    atributos = []

    # --- Leitura da imagem ---
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Inferência com YOLOv8 ---
    results = model(img_rgb)

    # --- Análise dos resultados ---
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id]

            # Filtro de confiança
            if conf > 0.5:
                if label == "person":
                    continue  # ignora pessoa (autoexplicativo)
                atributos.append(label)

    return atributos