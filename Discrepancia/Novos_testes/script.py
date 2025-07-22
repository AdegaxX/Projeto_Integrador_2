import cv2
import dlib
import face_recognition
import numpy as np
from imutils import face_utils
from numpy.linalg import norm

# === CONFIGURAÇÕES ===
predictor_path = r"/Discrepancia/Novos_testes/shape_predictor.dat"
imagem1_path = r"C:/Adegax/Ciência de dados - ADEGAS/5º semestre/Projeto integrador II/Imagens/rg/LeandroAdegas.jpg"
imagem2_path = r"C:/Users/AdegaX/Pictures/Camera Roll/WIN_20250721_17_04_51_Pro.jpg"

# === CARREGAR MODELOS ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# === FUNÇÃO: EXTRAIR LANDMARKS ===
def extrair_landmarks(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    rostos = detector(gray)

    if len(rostos) == 0:
        return None

    shape = predictor(gray, rostos[0])
    return face_utils.shape_to_np(shape)

# === FUNÇÃO: COMPARAR LANDMARKS ===
def comparar_landmarks(l1, l2):
    if l1 is None or l2 is None:
        return None
    return norm(l1 - l2)

# === ETAPA 1: EMBEDDINGS (IDENTIDADE) ===
img1_fr = face_recognition.load_image_file(imagem1_path)
img2_fr = face_recognition.load_image_file(imagem2_path)

enc1 = face_recognition.face_encodings(img1_fr)
enc2 = face_recognition.face_encodings(img2_fr)

if len(enc1) == 0 or len(enc2) == 0:
    print("❌ Não foi possível detectar rostos para embeddings.")
else:
    dist = norm(enc1[0] - enc2[0])
    print(f"🧬 Distância de embeddings (identidade): {dist:.4f}")
    if dist < 0.4:
        print("✅ Mesma pessoa (alta confiança)")
    elif dist < 0.6:
        print("🟡 Provavelmente a mesma pessoa")
    else:
        print("⚠️ Diferença significativa detectada")

# === ETAPA 2: LANDMARKS (FORMATO/ESTRUTURA FACIAL) ===
img1_cv = cv2.imread(imagem1_path)
img2_cv = cv2.imread(imagem2_path)

land1 = extrair_landmarks(img1_cv)
land2 = extrair_landmarks(img2_cv)

diff_landmarks = comparar_landmarks(land1, land2)
if diff_landmarks is not None:
    print(f"📏 Diferença entre landmarks: {diff_landmarks:.2f} px")
    if diff_landmarks < 10:
        print("✅ Estrutura facial praticamente idêntica")
    elif diff_landmarks < 25:
        print("🟡 Pequena mudança de expressão ou ângulo")
    else:
        print("⚠️ Diferença estrutural notável (posição, ângulo, expressão, cirurgia etc)")
else:
    print("❌ Não foi possível extrair landmarks de uma das imagens")

# === VISUALIZAÇÃO OPCIONAL ===
def desenhar_landmarks(imagem, pontos):
    for (x, y) in pontos:
        cv2.circle(imagem, (x, y), 1, (0, 255, 0), -1)
    return imagem

if land1 is not None and land2 is not None:
    img1_vis = desenhar_landmarks(img1_cv.copy(), land1)
    img2_vis = desenhar_landmarks(img2_cv.copy(), land2)

    cv2.imshow("Imagem 1 (RG)", img1_vis)
    cv2.imshow("Imagem 2 (Selfie)", img2_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# pip install "C:\Adegax\Ciência de dados - ADEGAS\5º semestre\Projeto integrador II\Projeto_Integrador_2\Discrepancia\Novos_testes\dlib-19.24.99-cp312-cp312-win_amd64.whl"