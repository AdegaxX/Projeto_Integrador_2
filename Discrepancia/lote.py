import os
from face_utils import get_embeddings, calculate_distance
from attribute_detector import detect_attributes
from evaluator import generate_report

rg_dir = "images/rg"
selfie_dir = "images/selfie"

rg_images = sorted([os.path.join(rg_dir, f) for f in os.listdir(rg_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
selfie_images = sorted([os.path.join(selfie_dir, f) for f in os.listdir(selfie_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

for rg_img, selfie_img in zip(rg_images, selfie_images):
    print(f"🔍 Analisando par: {os.path.basename(rg_img)} x {os.path.basename(selfie_img)}")

    try:
        emb_rg = get_embeddings(rg_img)
        emb_selfie = get_embeddings(selfie_img)
        distancia = calculate_distance(emb_rg, emb_selfie)
        atributos = detect_attributes(selfie_img)
        relatorio = generate_report(distancia, atributos)
        print(relatorio)
        print("=" * 60)
    except Exception as e:
        print(f"❌ Erro ao analisar {rg_img} x {selfie_img}: {e}")
