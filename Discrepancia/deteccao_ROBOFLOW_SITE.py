# pip install inference supervision opencv-python

from inference import get_model
import supervision as sv
import cv2

# Substitua pelo seu model_id e API_KEY
MODEL_ID = "detec_objetos/5"
API_KEY = "mp6z8hktgskSRsKdXT8Y"    # não compartilhar com ninguém

# Carrega o modelo com autenticação
model = get_model(model_id=MODEL_ID, api_key=API_KEY)

# Webcam + detecção
cap = cv2.VideoCapture(0)
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)

    annotated = bounding_box_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections)

    cv2.imshow("Webcam Roboflow", annotated)
    if cv2.waitKey(1) & 0xFF == 27:     # tecla Esc para sair
        break

cap.release()
cv2.destroyAllWindows()
