from ultralytics import RTDETR

if __name__ == '__main__':
    # Load a model
    model = RTDETR("cfg/model/rtdetr-resnet50.yaml")  # build a new model from scratch
    # Use the model
    model.train(data = "SDC_dataset/dataset.yaml",cfg="cfg/default.yaml")  # train the model
