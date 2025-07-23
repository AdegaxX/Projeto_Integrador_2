import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference import get_model

# Detecta e recorta a maior face
def detectar_maior_face(image_np, model, margin=40):
    result = model.infer(image_np, confidence=0.7)
    detections = sv.Detections.from_inference(result[0])

    faces = []
    for bbox, label in zip(detections.xyxy, result[0].predictions):
        if label.class_name.lower() == "face":
            x1, y1, x2, y2 = map(int, bbox)
            area = (x2 - x1) * (y2 - y1)
            faces.append((area, (x1, y1, x2, y2)))

    if faces:
        _, (x1, y1, x2, y2) = max(faces, key=lambda x: x[0])
        h, w, _ = image_np.shape
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        return image_np[y1:y2, x1:x2]
    return None

# Desenha todas as detecções e retorna as classes
def desenhar_deteccoes(image_np, model):
    image_copy = image_np.copy()
    result = model.infer(image_np, confidence=0.7)
    detections = sv.Detections.from_inference(result[0])
    classes_detectadas = []

    for bbox, label in zip(detections.xyxy, result[0].predictions):
        x1, y1, x2, y2 = map(int, bbox)
        classe = label.class_name
        classes_detectadas.append(classe)

        color = {
            "face": (0, 255, 0),
            "hat": (255, 255, 0),
            "glass": (255, 0, 255),
            "bar": (0, 165, 255),
            "mask": (255, 0, 0)
        }.get(classe.lower(), (200, 200, 200))

        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            image_copy,
            classe,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            4
        )
    return image_copy, classes_detectadas

# Streamlit
st.set_page_config(page_title="Verificador de RG", layout="centered")
st.title("Discrepância entre RG e SELFIE")
st.markdown("Envie duas imagens da mesma pessoa e descubra se a mudança facial exige refazer o RG.")

img1 = st.file_uploader("Envie uma foto do RG", type=["jpg", "jpeg", "png"])
img2 = st.file_uploader("Envie uma foto atual", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return get_model(model_id="detec_objetos/6", api_key="mp6z8hktgskSRsKdXT8Y")

model = load_model()

if img1 and img2:
    img_rg = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_COLOR)
    img_nova = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_COLOR)

    face1 = detectar_maior_face(img_rg, model)
    face2 = detectar_maior_face(img_nova, model)

    if face1 is not None and face2 is not None and face1.shape[0] > 20 and face2.shape[0] > 20:
        st.subheader("Comparando as imagens com DeepFace")

        cv2.imwrite("temp1.jpg", face1)
        cv2.imwrite("temp2.jpg", face2)

        try:
            import sys
            import subprocess
            import time
            import os

            subprocess.run([sys.executable, r"C:\Adegax\Ciência de dados - ADEGAS\5º semestre\Projeto integrador II\Projeto_Integrador_2\Discrepancia\deepFace.py"])

            output_file = "deepface_output.txt"
            timeout = 5
            start = time.time()
            while not os.path.exists(output_file):
                if time.time() - start > timeout:
                    st.error("Timeout: O arquivo de saída não foi gerado.")
                    break
                time.sleep(0.1)

            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    output = f.read()
                st.subheader("Resultado da Comparação:")
                st.code(output)

                # Só desenha e mostra a imagem da selfie (foto atual)
                img2_boxed, classes2 = desenhar_deteccoes(img_nova, model)
                st.subheader("Detecções na Foto Atual:")
                st.image(cv2.cvtColor(img2_boxed, cv2.COLOR_BGR2RGB), channels="RGB")

                acessorios = [c for c in classes2 if c.lower() != "face"]
                if acessorios:
                    st.markdown("**Acessórios detectados:**")
                    st.info(", ".join(set(acessorios)))
                else:
                    st.success("Nenhum acessório detectado.")

        except Exception as e:
            st.error(f"Erro ao comparar rostos: {e}")
    else:
        st.error("Não foi possível detectar um rosto válido nas imagens.")
