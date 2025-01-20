from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"D:\BuffDetect\pose\train8\train8\weights\best.pt")  # load a pretrained model (recommended for training)
    success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
    assert success
