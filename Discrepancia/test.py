import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from deepface import DeepFace

# Função: detecta o rosto com base no label 'face'
def detect_face(image_np, model, margin=40):
    result = model.infer(image_np, confidence=0.3)
    detections = sv.Detections.from_inference(result[0])

    # Inicializa listas para limites combinados
    x1s, y1s, x2s, y2s = [], [], [], []

    for bbox, label in zip(detections.xyxy, result[0].predictions):
        print(f"📦 Detecção: {label.class_name}")
        # Considera qualquer classe como parte do rosto
        x1, y1, x2, y2 = map(int, bbox)
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

    if x1s and y1s and x2s and y2s:
        # Combina os limites para formar um retângulo único
        x1 = max(0, min(x1s) - margin)
        y1 = max(0, min(y1s) - margin)
        x2 = min(image_np.shape[1], max(x2s) + margin)
        y2 = min(image_np.shape[0], max(y2s) + margin)

        return image_np[y1:y2, x1:x2]

    return None

# Configurações do app
st.set_page_config(page_title="Verificador de RG", layout="centered", page_icon="🪪")
st.title("🪪 Verificador de RG por Face Match")
st.markdown("Envie duas imagens da mesma pessoa e descubra se a mudança facial exige refazer o RG.")

# Upload das imagens
img1 = st.file_uploader("📷 Foto do RG", type=["jpg", "jpeg", "png"])
img2 = st.file_uploader("🤳 Foto Atual", type=["jpg", "jpeg", "png"])

# Carrega modelo do Roboflow
@st.cache_resource
def load_model():
    return get_model(model_id="detec_objetos/5", api_key="mp6z8hktgskSRsKdXT8Y")

model = load_model()
print(model)
# Quando ambas forem enviadas
if img1 and img2:
    # Converte para imagem OpenCV
    img_rg = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_COLOR)
    img_nova = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_COLOR)

   # st.subheader("🔍 Detectando rostos...")

    face1 = detect_face(img_rg, model)
    face2 = detect_face(img_nova, model)

    if face1 is not None and face2 is not None:

        st.subheader("🤖 Comparando com DeepFace...")

        # Salva temporariamente
        cv2.imwrite("temp1.jpg", face1)
        cv2.imwrite("temp2.jpg", face2)

        try:
            # E chama o verificador externo:
            import sys
            import subprocess
            import time
            import os
            subprocess.run([sys.executable, r"C:\Adegax\Ciência de dados - ADEGAS\5º semestre\Projeto integrador II\Projeto_Integrador_2\Discrepancia\deepTest.py"])

            # Espera o arquivo ser gerado
            output_file = "deepface_output.txt"
            timeout = 5  # segundos
            start = time.time()
            while not os.path.exists(output_file):
                if time.time() - start > timeout:
                    st.error("❌ Timeout: O arquivo de saída não foi gerado.")
                    break
                time.sleep(0.1)

            # Lê e exibe a saída
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    output = f.read()
                st.subheader("📤 Saída do DeepFace:")
                st.code(output)

        except Exception as e:
            st.error(f"Erro ao comparar rostos: {e}")
    else:
        st.error("Não foi possível detectar o rosto em uma das imagens.")
