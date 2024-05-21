from ultralytics import RTDETR

if __name__ == '__main__':
    # Load a model
    model = RTDETR("D:/vision-deteccion-objetos/cfg/model/rtdetr-resnet50.yaml")  # build a new model from scratch
    # Use the model
    model.train(data = "D:/SDC_dataset/dataset.yaml",cfg="D:/vision-deteccion-objetos/cfg/default.yaml")  # train the model
