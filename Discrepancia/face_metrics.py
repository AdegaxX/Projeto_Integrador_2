from deepface import DeepFace
import cv2

def extract_face_embedding(img_path):
    return DeepFace.represent(img_path=img_path, model_name="Facenet512", enforce_detection=True)[0]['embedding']

def compare_faces(img1, img2):
    emb1 = extract_face_embedding(img1)
    emb2 = extract_face_embedding(img2)
    from numpy.linalg import norm
    distance = norm(emb1 - emb2)
    return distance