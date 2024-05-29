from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r"D:\vision-deteccion-objetos\UDIT_test\train13\weights\best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict(r"D:\SDC_dataset\images\train\1478019952686311006_jpg.rf.8AtteKyLJUNzZ2fEpxRj.jpg", save=True, imgsz=640, conf=0.5)