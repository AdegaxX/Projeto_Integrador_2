import cv2
from deepface import DeepFace

# Caminhos absolutos para evitar problemas com acentuação
FACE_CASCADE_PATH = "C:/Adegax/haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = "C:/Adegax/haarcascade_eye.xml"

# Carrega os classificadores Haarcascade
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)


def detect_attributes(img_path):
    atributos = []

    # --- Análise com DeepFace ---
    try:
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )

        analise = result[0] if isinstance(result, list) else result
        idade = analise.get("age", 0)
        genero = analise.get("gender", "")

        # Heurística simples para barba
        if genero == "Man" and idade >= 20:
            atributos.append("Barba (suposta)")

    except Exception as e:
        print(f"[Erro] Análise de atributos falhou (DeepFace): {e}")

    # --- Detecção com OpenCV para óculos ---
    try:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 0:
                atributos.append("Óculos escuros ou obstrução nos olhos")
            elif len(eyes) == 1:
                atributos.append("Possível óculos ou obstrução parcial")

            break  # Analisa apenas o primeiro rosto encontrado

    except Exception as e:
        print(f"[Erro] Análise de atributos falhou (OpenCV): {e}")

    return atributos
