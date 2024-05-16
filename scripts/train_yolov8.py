from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("cfg/model/yolov8m.yaml")  # build a new model from scratch
    # Use the model
    model.train(data = r"SDC_dataset/dataset.yaml",cfg=r"cfg/default.yaml")  # train the model
