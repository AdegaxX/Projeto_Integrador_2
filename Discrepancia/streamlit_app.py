# Para abrir:
# Digite no terminal: & "C:\Adegax\Ciência de dados - ADEGAS\5º semestre\Mineração de dados\.venv\Scripts\Activate.ps1"
# Depois quando estiver no (.venv) digite: streamlit run streamlit_app.py



import streamlit as st
from PIL import Image
import os
import tempfile

from face_utils import get_embeddings, calculate_distance
from attribute_detector import detect_attributes
from evaluator import avaliar_discrepancia

st.set_page_config(page_title="Analisador de Discrepância Facial", layout="centered")

st.title("📸 Analisador de Discrepância entre RG e Selfie")

st.markdown("Envie a foto do RG e uma selfie atual para verificar se você precisa atualizar a foto no documento ou remover acessórios.")

# Uploads
rg_image = st.file_uploader("📄 Envie a imagem do RG", type=["jpg", "jpeg", "png"])
selfie_image = st.file_uploader("🤳 Envie sua selfie atual", type=["jpg", "jpeg", "png"])

if rg_image and selfie_image:
    with tempfile.TemporaryDirectory() as tmpdir:
        rg_path = os.path.join(tmpdir, "rg.jpg")
        selfie_path = os.path.join(tmpdir, "selfie.jpg")

        with open(rg_path, "wb") as f:
            f.write(rg_image.read())
        with open(selfie_path, "wb") as f:
            f.write(selfie_image.read())

        st.image([rg_path, selfie_path], caption=["Imagem do RG", "Selfie atual"], width=250)

        try:
            emb_rg = get_embeddings(rg_path)
            emb_selfie = get_embeddings(selfie_path)

            distancia = calculate_distance(emb_rg, emb_selfie)
            atributos = detect_attributes(selfie_path)
            recomendacao = avaliar_discrepancia(distancia, atributos)

            st.markdown("### Resultado da Análise")
            st.write(f"**Distância facial**: `{distancia:.4f}`")
            st.write(f"**Acessórios detectados**: `{atributos}`")
            st.success(f"**Recomendação**: {recomendacao}")

        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")
else:
    st.info("Por favor, envie as duas imagens para iniciar a análise.")
