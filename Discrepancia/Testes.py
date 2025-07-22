import streamlit as st
import numpy as np
import cv2
import tempfile
import os

from insightface.app import FaceAnalysis
import supervision as sv
from inference import get_model

# Configuração do modelo InsightFace
insight_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
insight_app.prepare(ctx_id=0)

# Streamlit setup
st.set_page_config(page_title="Análise de Discrepância Facial", layout="centered")
st.title("📸 Analisador de RG vs Selfie – Acessórios + Discrepância Facial")

# Upload de imagens
rg_file = st.file_uploader("Envie a imagem do RG", type=["jpg", "jpeg", "png"])
selfie_file = st.file_uploader("Envie sua selfie atual", type=["jpg", "jpeg", "png"])

# Caminho e API do modelo de detecção de acessórios
MODEL_ID = "detec_objetos/5"
API_KEY = "mp6z8hktgskSRsKdXT8Y"

def redimensionar_e_salvar(path_or_bytes, label):
    if isinstance(path_or_bytes, bytes):
        npimg = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path_or_bytes)

    if img is None:
        return None, None

    max_dim = max(img.shape[:2])
    if max_dim > 1080:
        scale = 1080 / max_dim
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix=f"{label}_")
    cv2.imwrite(temp.name, img)
    return img, temp.name

def detectar_e_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None

    faces = insight_app.get(img)
    if not faces:
        return None, None, None

    # Seleciona a face de maior área
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    face_crop = face.crop_img
    embedding = face.embedding  # vetor 512D
    bbox = face.bbox.astype(int)
    return face_crop, embedding, bbox

def mostrar_face(img_path, bbox, label):
    img = cv2.imread(img_path)
    if img is None or bbox is None:
        st.write(f"❌ Não foi possível exibir a face de: {label}")
        return
    x1, y1, x2, y2 = bbox
    img_box = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
    st.image(img_box, caption=f"{label} (InsightFace)", use_container_width=True)

# Lógica principal
if rg_file and selfie_file:
    img_rg, rg_path = redimensionar_e_salvar(rg_file.read(), "rg")
    img_selfie, selfie_path = redimensionar_e_salvar(selfie_file.read(), "selfie")

    st.image([img_rg, img_selfie], caption=["Imagem RG", "Imagem Selfie"], width=250)

    st.markdown("---")
    st.subheader("🧾 Resultado da Análise")

    rg_crop, emb_rg, bbox_rg = detectar_e_embedding(rg_path)
    selfie_crop, emb_selfie, bbox_selfie = detectar_e_embedding(selfie_path)

    if emb_rg is None or emb_selfie is None:
        st.error("❌ Não foi possível gerar os embeddings faciais.")
        st.markdown("### 👤 Tentativa de visualização das regiões faciais detectadas:")
        mostrar_face(rg_path, bbox_rg, "Imagem RG")
        mostrar_face(selfie_path, bbox_selfie, "Imagem Selfie")
    else:
        distancia = np.linalg.norm(emb_rg - emb_selfie)

        if distancia > 0.7:
            rec = "⚠️ Discrepância alta – Recomenda-se atualizar a foto."
        elif distancia > 0.45:
            rec = "🟡 Discrepância leve – Verifique uso de acessórios ou mudanças faciais."
        else:
            rec = "✅ Rosto consistente – Foto válida."

        st.write(f"**Distância facial:** `{distancia:.4f}`")

        # Detecção de acessórios com API externa
        model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        results = model.infer(selfie_path)[0]
        detections = sv.Detections.from_inference(results)
        classes_detectadas = detections.data['class_name']

        st.write(f"**Acessórios detectados:** {', '.join(classes_detectadas) if classes_detectadas else 'Nenhum'}")
        st.success(f"**Recomendação:** {rec}")
