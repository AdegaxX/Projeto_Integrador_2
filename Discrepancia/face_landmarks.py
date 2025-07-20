import face_recognition
import cv2


def get_face_metrics(img_path):
    image = face_recognition.load_image_file(img_path)
    face_landmarks = face_recognition.face_landmarks(image)

    if not face_landmarks:
        return None

    landmarks = face_landmarks[0]
    nose = landmarks['nose_bridge']
    eyes = landmarks['left_eye'] + landmarks['right_eye']
    chin = landmarks['chin']

    face_height = chin[-1][1] - nose[0][1]
    face_width = eyes[-1][0] - eyes[0][0]

    return {
        'altura_face_px': face_height,
        'largura_face_px': face_width,
        'proporcao': face_width / face_height if face_height != 0 else None
    }