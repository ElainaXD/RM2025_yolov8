import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # path to the YOLO model
source_folder = r"/app/Desktop/yolov7/police"  # path to the source folder
crop_root_folder = r"/app/Desktop/yolov7/police/crop"  # path to the root folder for saving crops

if not os.path.exists(source_folder):
    print("Source folder not found")
    sys.exit()

if not os.path.exists(crop_root_folder):
    os.makedirs(crop_root_folder)

def save_cropped_images(source_path, crop_path, frame, boxes):
    crop_count = 0
    for box in boxes:
        crop_count += 1
        crop_object = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        crop_filename = f"crop_{crop_count}.png"
        cv2.imwrite(os.path.join(crop_path, crop_filename), crop_object)

def process_images(source_folder, crop_root_folder):
    for subdir, dirs, files in os.walk(source_folder):
        crop_subdir = subdir.replace(source_folder, crop_root_folder)
        if not os.path.exists(crop_subdir):
            os.makedirs(crop_subdir)

        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                frame = cv2.imread(filepath)
                results = model.predict(frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu()
                save_cropped_images(subdir, crop_subdir, frame, boxes)

process_images(source_folder, crop_root_folder)
