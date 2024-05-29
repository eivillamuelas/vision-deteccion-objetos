from ultralytics import YOLOv10, YOLO


if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLOv10("yolov10m.pt")
    model8 = YOLO("yolov8n.pt")

    # Perform tracking with the model
    results = model.track(r"D:\vision-deteccion-objetos\img\Traffic_top_view.mp4", show=True, tracker="cfg/bytetrack.yaml", imgsz=640, conf=0.1, iou=0.1)  # Tracking with default tracker