from ultralytics import YOLOv10


if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLOv10("yolov10m.pt")

    # Run inference on 'bus.jpg' with arguments
    model.predict([r"D:\SDC_dataset\images\train\1478019952686311006_jpg.rf.8AtteKyLJUNzZ2fEpxRj.jpg", r"D:\SDC_dataset\images\val\1478020228190773357_jpg.rf.IlN1giepmV9XVXSlR4uz.jpg"], save=True, imgsz=640, conf=0.1, iou=0.4)