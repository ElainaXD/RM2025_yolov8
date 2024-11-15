import cv2
from ultralytics import YOLO
import numpy as np
import torch
import os

# Set the device to GPU if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the YOLO model on the specified device
#model = YOLO(r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\weights\runs\detect\train20\weights\best.pt").to(device)
model = YOLO(r"/app/weights/best.pt").to(device)

# Path to the input video
#input_path = r"app/videos\crowding_dispersal_6ppl_010.mp4"
input_path = r"/app/videos/F2PerimeterFenceP2-10CAM01_throwing.mp4"


# Open the source video
cap = cv2.VideoCapture(input_path)

# Extract the base name of the input file and create the output file name
base_name = os.path.basename(input_path)
#output_path = os.path.join(r'\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\video_output', base_name)
#output_path = os.path.join(r'/app/video_output', base_name)

# Define the codec and create a VideoWriter object for AVI format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = os.path.join(r'/app/video_output', os.path.splitext(base_name)[0] + '.avi')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

try:
    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO prediction with specified confidence level
        results = model(frame_rgb, conf=0.2, classes=0)

        # Customize and plot results
        for r in results:
            im_array = r.plot(line_width=2, font_size=2)
            # Convert array from RGB (PIL format) to BGR (OpenCV format)
            frame_processed = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

        # Write the processed frame to the output video
        out.write(frame_processed)

finally:
    # Release resources
    cap.release()
    out.release()
    print("completed!")
