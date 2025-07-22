import cv2
from ultralytics import YOLO

# Carrega modelo YOLOv8 treinado
model = YOLO("runs/detect/train/weights/best.pt")

# Carrega classificador HaarCascade para detecção de rosto
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Certifique-se de ter o arquivo no mesmo diretório

# Função para detectar rostos e retornar coordenadas
def detectar_rosto_cv2(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return rostos  # Lista de (x, y, w, h)

# Inicia webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === YOLOv8: detecção de acessórios ===
    results = model(frame, conf=0.05)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            label_text = f"{label} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # === OpenCV: detecção de rosto ===
    rostos = detectar_rosto_cv2(frame)

    if len(rostos) > 0:
        rosto_label = "ROSTO DETECTADO ✅"
        for (x, y, w, h) in rostos:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    else:
        rosto_label = "ROSTO NÃO DETECTADO ❌"

    # Texto fixo no topo
    cv2.putText(frame, rosto_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # Exibe imagem
    cv2.imshow("Detecção de Acessórios + Rosto (YOLOv8 + OpenCV)", frame)

    # Sai com ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
