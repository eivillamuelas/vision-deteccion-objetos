from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("D:/vision-deteccion-objetos/yolov10m.pt")  # build a new model from scratch
    # Use the model
    model.train(data = "D:/SDC_dataset/dataset.yaml",cfg="D:/vision-deteccion-objetos/cfg/default.yaml")  # train the model
