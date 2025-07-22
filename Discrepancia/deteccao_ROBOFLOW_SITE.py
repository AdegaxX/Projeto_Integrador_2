from roboflow import Roboflow
import supervision as sv
from supervision.draw.color import Color
from supervision.draw.box_annotator import BoxAnnotator
import cv2

# Inicializa o projeto Roboflow
print("loading Roboflow workspace...")
rf = Roboflow(api_key="mp6z8hktgskSRsKdXT8Y")  # Substitua aqui
print("loading Roboflow project...")
project = rf.workspace("adegax").project("detec_objetos")
model = project.version(5).model  # Exemplo: .version(1)

# Caminho da imagem
image_path = "https://images.pexels.com/photos/2745944/pexels-photo-2745944.jpeg"
image = cv2.imread(image_path)

# Faz a predição
result = model.predict(image_path).json()

# Extrai as detecções
xyxy = []
class_id = []
confidence = []

for pred in result["predictions"]:
    x1 = int(pred["x"] - pred["width"] / 2)
    y1 = int(pred["y"] - pred["height"] / 2)
    x2 = int(pred["x"] + pred["width"] / 2)
    y2 = int(pred["y"] + pred["height"] / 2)
    xyxy.append([x1, y1, x2, y2])
    class_id.append(project.classes.index(pred["class"]))
    confidence.append(pred["confidence"])

# Cria as detecções
detections = sv.Detections(
    xyxy=xyxy,
    class_id=class_id,
    confidence=confidence
)

# Define a ferramenta de anotação
box_annotator = BoxAnnotator(color=Color.red(), thickness=2, text_thickness=1, text_scale=0.5)

# Anota a imagem
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    labels=[f"{project.classes[c]} {conf:.2f}" for c, conf in zip(class_id, confidence)]
)

# Salva a imagem
cv2.imwrite("resultado_roboflow.jpg", annotated_image)
print("Imagem anotada salva como resultado_roboflow.jpg")
