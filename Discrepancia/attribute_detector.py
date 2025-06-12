# Funções para identificar atributos (barba, óculos, etc.)

from deepface import DeepFace


def detect_attributes(img_path):
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion', 'facial_feature'], enforce_detection=False)   # Se você confia que a imagem tem um rosto, mas o DeepFace está sendo exigente demais, pode desativar enforce_detection:
        atributos = []

        # Exemplo: detectando barba, óculos ficticiamente
        if 'glasses' in result.get('facial_feature', {}):
            atributos.append("Óculos")
        if result[0]["gender"] == "Man" and result[0]["age"] > 18:
            atributos.append("barba")  # isso é só um exemplo
        # Você pode usar heurísticas visuais ou bibliotecas como mediapipe para detalhes melhores
        return atributos
    except Exception as e:
        print(f"[Erro] Análise de atributos falhou: {e}")
        return []
