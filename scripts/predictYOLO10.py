from ultralytics import YOLOv10


if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLOv10(r"D:\vision-deteccion-objetos\UDIT_test\train13\weights\best.pt")

    # Run inference on 'bus.jpg' with arguments
    model.predict(r"D:\vision-deteccion-objetos\img\LA.mp4", save=True, imgsz=640, conf=0.5, iou=0.4)