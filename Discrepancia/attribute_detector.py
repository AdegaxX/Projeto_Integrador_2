import cv2
from ultralytics import YOLO

# Carrega modelo custom para acessórios faciais
# model = YOLO('acessorios.pt')  # Usar esse se eu quiser treinar o modelo para detectar outras coisas

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Dados gerados pelo treinamento YOLO:
model = YOLO('runs/detect/train/weights/best.pt')


def detect_attributes(img_path):
    atributos = []
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id]
            if conf > 0.5:
                atributos.append(label)
    return atributos