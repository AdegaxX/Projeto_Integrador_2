import streamlit as st
from deepface import DeepFace
from deepface.modules import detection
import numpy as np
import cv2
import tempfile
from inference import get_model
import supervision as sv
import os

MODEL_ID = "detec_objetos/5"
API_KEY = "mp6z8hktgskSRsKdXT8Y"
DETECTOR_BACKENDS = ["retinaface", "mtcnn", "opencv"]

st.set_page_config(page_title="Análise de Discrepância Facial", layout="centered")
st.title("📸 Analisador de RG vs Selfie – Acessórios + Discrepância Facial")

rg_file = st.file_uploader("Envie a imagem do RG", type=["jpg", "jpeg", "png"])
selfie_file = st.file_uploader("Envie sua selfie atual", type=["jpg", "jpeg", "png"])

def redimensionar_e_salvar(path_or_bytes, label):
    # Carrega imagem de arquivo ou bytes e salva redimensionada
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

def recortar_face(img_path):
    for backend in DETECTOR_BACKENDS:
        try:
            faces = detection.extract_faces(
                img_path=img_path,
                detector_backend=backend,
                enforce_detection=True
            )
            if faces and isinstance(faces[0], tuple):
                return faces[0][0]
        except Exception as e:
            st.warning(f"[{backend}] falhou: {e}")
    return None

def get_embedding_from_crop(crop_img):
    try:
        result = DeepFace.represent(crop_img, model_name="Facenet512", enforce_detection=False)
        if isinstance(result, list) and len(result) > 0 and 'embedding' in result[0]:
            return np.array(result[0]['embedding'])
    except:
        return None
    return None

def mostrar_rosto_detectado(img_path, label):
    img = cv2.imread(img_path)
    for backend in DETECTOR_BACKENDS:
        try:
            faces = detection.extract_faces(
                img_path=img_path,
                detector_backend=backend,
                enforce_detection=True
            )
            if faces and isinstance(faces[0], tuple):
                (x, y, w, h) = faces[0][1]
                img_box = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
                st.image(img_box, caption=f"{label} (detectado com {backend})", use_container_width=True)
                return
        except:
            continue
    st.write(f"❌ Não foi possível exibir a face detectada em: {label}")

if rg_file and selfie_file:
    img_rg, rg_path = redimensionar_e_salvar(rg_file.read(), "rg")
    img_selfie, selfie_path = redimensionar_e_salvar(selfie_file.read(), "selfie")

    st.image([img_rg, img_selfie], caption=["Imagem RG", "Imagem Selfie"], width=250)

    st.markdown("---")
    st.subheader("🧾 Resultado da Análise")

    rg_face = recortar_face(rg_path)
    selfie_face = recortar_face(selfie_path)

    if rg_face is None or selfie_face is None:
        st.error("❌ Um ou ambos os embeddings não puderam ser gerados.")
        st.markdown("### 👤 Tentativa de visualização das regiões faciais detectadas:")
        mostrar_rosto_detectado(rg_path, "Imagem RG")
        mostrar_rosto_detectado(selfie_path, "Imagem Selfie")
    else:
        emb_rg = get_embedding_from_crop(rg_face)
        emb_selfie = get_embedding_from_crop(selfie_face)

        if emb_rg is None or emb_selfie is None:
            st.error("❌ Erro ao gerar os vetores faciais.")
        else:
            distancia = np.linalg.norm(emb_rg - emb_selfie)

            if distancia > 0.7:
                rec = "⚠️ Discrepância alta – Recomenda-se atualizar a foto."
            elif distancia > 0.45:
                rec = "🟡 Discrepância leve – Verifique uso de acessórios ou mudanças faciais."
            else:
                rec = "✅ Rosto consistente – Foto válida."

            st.write(f"**Distância facial:** `{distancia:.4f}`")

            model = get_model(model_id=MODEL_ID, api_key=API_KEY)
            results = model.infer(selfie_path)[0]
            detections = sv.Detections.from_inference(results)
            classes_detectadas = detections.data['class_name']

            st.write(f"**Acessórios detectados:** {', '.join(classes_detectadas) if classes_detectadas else 'Nenhum'}")
            st.success(f"**Recomendação:** {rec}")
