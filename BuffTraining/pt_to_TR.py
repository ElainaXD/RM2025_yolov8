from ultralytics import YOLO

# Load a model
model = YOLO(r"D:\BuffDetect\pose\train8\train8\weights\best.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=0)  # export the model to engine format
assert success
