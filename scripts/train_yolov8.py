from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("D:/vision-deteccion-objetos/cfg/model/yolov8.yaml")  # build a new model from scratch
    # Use the model
    model.train(data = r"D:/SDC_dataset/dataset.yaml",cfg=r"D:/vision-deteccion-objetos/cfg/default.yaml")  # train the model
