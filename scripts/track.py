from ultralytics import YOLOv10


if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLOv10(r"D:\vision-deteccion-objetos\UDIT_test\train13\weights\best.pt")

    # Perform tracking with the model
    results = model.track(r"D:\vision-deteccion-objetos\img\LA.mp4", show=True, tracker="cfg/bytetrack.yaml", imgsz=640, conf=0.5, iou=0.4)  # Tracking with default tracker