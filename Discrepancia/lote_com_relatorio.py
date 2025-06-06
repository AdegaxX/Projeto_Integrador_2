import os
from face_utils import get_embeddings, calculate_distance
from attribute_detector import detect_attributes
from evaluator import generate_report

rg_dir = "images/rg"
selfie_dir = "images/selfie"
output_dir = "relatorios"

os.makedirs(output_dir, exist_ok=True)

rg_files = {f.lower(): os.path.join(rg_dir, f) for f in os.listdir(rg_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
selfie_files = {f.lower(): os.path.join(selfie_dir, f) for f in os.listdir(selfie_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}

# Casar por nome
pessoas = set(rg_files.keys()) & set(selfie_files.keys())

for pessoa in pessoas:
    print(f"🔍 Analisando: {pessoa}")
    rg_img = rg_files[pessoa]
    selfie_img = selfie_files[pessoa]

    try:
        emb_rg = get_embeddings(rg_img)
        emb_selfie = get_embeddings(selfie_img)
        distancia = calculate_distance(emb_rg, emb_selfie)
        atributos = detect_attributes(selfie_img)
        relatorio = generate_report(distancia, atributos)

        print(relatorio)
        print("=" * 60)

        # Salvar relatório em arquivo
        nome_base = os.path.splitext(pessoa)[0]
        with open(os.path.join(output_dir, f"relatorio_{nome_base}.txt"), "w") as f:
            f.write(relatorio)

    except Exception as e:
        print(f"❌ Erro ao analisar {pessoa}: {e}")
