# Funções para detecção e embeddings faciais

from deepface import DeepFace
import numpy as np

def get_embeddings(img_path):
    obj = DeepFace.represent(img_path=img_path, model_name="Facenet512", enforce_detection=False)      # Se você confia que a imagem tem um rosto, mas o DeepFace está sendo exigente demais, pode desativar enforce_detection:

    return np.array(obj[0]["embedding"])

def calculate_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)



