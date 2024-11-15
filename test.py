from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('.//yolov8x.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('./output1', save=True,)

# success = model.export(format='onnx')
