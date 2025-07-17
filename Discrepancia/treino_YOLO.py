from ultralytics import YOLO

model = YOLO('yolov8n.pt')

modelo = "data.yaml"

model.train(
    data=modelo,
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu'
)
