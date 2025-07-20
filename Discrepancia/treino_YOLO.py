from ultralytics import YOLO

model = YOLO('yolov8n.pt')

modelo = r"C:\Adegax\Ciência de dados - ADEGAS\5º semestre\Projeto integrador II\Projeto_Integrador_2\Discrepancia\treioROBOFLOW\data.yaml"

model.train(
    data=modelo,
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu'
)
