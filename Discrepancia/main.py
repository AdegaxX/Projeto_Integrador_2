# Arquivo principal de execução

from face_utils import get_embeddings, calculate_distance
from attribute_detector import detect_attributes
from evaluator import generate_report

# Caminhos das imagens
foto_rg = "images/rg.jpg"
foto_atual = "images/selfie.jpg"

# Etapa 1: Gera embeddings faciais
embedding_rg = get_embeddings(foto_rg)
embedding_selfie = get_embeddings(foto_atual)

# Etapa 2: Calcula distância entre as duas imagens
discrepancia = calculate_distance(embedding_rg, embedding_selfie)

# Etapa 3: Detecta atributos faciais na selfie
atributos = detect_attributes(foto_atual)

# Etapa 4: Gera e imprime relatório final
relatorio = generate_report(discrepancia, atributos)
print(relatorio)