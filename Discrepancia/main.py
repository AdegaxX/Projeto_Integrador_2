from attribute_detector import detect_attributes
from face_metrics import compare_faces
from face_landmarks import get_face_metrics

img_rg = "images/rg.jpg"
img_selfie = "images/selfie.jpg"

atributos = detect_attributes(img_selfie)
distancia = compare_faces(img_rg, img_selfie)
metricas = get_face_metrics(img_selfie)

print(f"Acessórios detectados: {atributos}")
print(f"Distância facial (vetorial): {distancia:.4f}")
print(f"Métricas faciais: {metricas}")