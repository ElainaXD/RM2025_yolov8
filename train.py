from ultralytics import YOLO 
import os
model = YOLO('last.pt')

if __name__ == '__main__':
    os.environ['WANDB_MODE'] = 'offline'
    results = model.train(data='cynapse.yaml', epochs=100, save=True, save_period=1,batch=12, imgsz=640, device=0,workers=16, resume = True)
    results = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = YOLO("yolov8.pt").export(format="onnx")  # export a model to ONNX format
