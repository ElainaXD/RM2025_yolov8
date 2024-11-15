import cv2
import numpy as np
import os
import random
import datetime

# Class ID to label mapping
class_id_to_label = {
    0: 'person',
    1: 'head'
}

def load_image(image_path):
    """Load an image from a file path."""
    return cv2.imread(image_path)

def load_labels_yolo(label_path, img_width, img_height):
    """Load bounding box labels from a file path in YOLO format and denormalize."""
    boxes = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            boxes.append((int(class_id), x_min, y_min, x_max, y_max))
    return boxes


def normalize_boxes(boxes, img_width, img_height):
    """Normalize bounding box coordinates for the mosaic."""
    normalized_boxes = []
    for box in boxes:
        class_id, x_min, y_min, x_max, y_max = box
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        normalized_boxes.append((class_id, x_center, y_center, width, height))
    return normalized_boxes


def save_image(image, output_folder, img_index):
    """Save an image to a specified folder with a unique filename."""
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"mosaic_{img_index}_{timestamp}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, image)

# Example function that uses the mapping
def save_labels(boxes, output_folder, img_index):
    """Save normalized bounding box labels to a file."""
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"mosaic_{img_index}_{timestamp}.txt"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, 'w') as file:
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            # Save the class ID and coordinates directly without converting to a label name
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
def adjust_boxes(boxes, dx, dy, scale):
    """Adjust bounding box coordinates for the mosaic."""
    adjusted_boxes = []
    for box in boxes:
        class_id, x_min, y_min, x_max, y_max = box  # Include class_id in the unpacking
        x_min, x_max = int(x_min * scale + dx), int(x_max * scale + dx)
        y_min, y_max = int(y_min * scale + dy), int(y_max * scale + dy)
        adjusted_boxes.append((class_id, x_min, y_min, x_max, y_max))  # Include class_id in the adjusted box
    return adjusted_boxes


def create_mosaic(images, boxes, img_size=640):
    """Create a 2x2 mosaic image and adjust bounding boxes."""
    assert len(images) == 4, "Need exactly 4 images for a 2x2 mosaic."
    mosaic_img = np.zeros((img_size * 2, img_size * 2, 3), dtype=np.uint8)
    new_boxes = []

    for i, (img, box) in enumerate(zip(images, boxes)):
        h, w = img.shape[:2]
        scale = min(img_size / h, img_size / w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        dx, dy = i % 2 * img_size, i // 2 * img_size
        mosaic_img[dy:dy + int(h * scale), dx:dx + int(w * scale)] = img

        adjusted_box = adjust_boxes(box, dx, dy, scale)
        new_boxes.extend(adjusted_box)

    return mosaic_img, new_boxes


# Example Usage
image_folder = r"/app/data_v3/images/val/CrowdHuman1"
label_folder = r"/app/data_v3/labels/val/CrowdHuman1"
output_folder = r"/app/data_v3/images/train/mosaic"

# # Example Usage
# image_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v3\images\val\c146"
# label_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v3\labels\val\c146"
# output_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\augmented"

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
label_paths = [os.path.join(label_folder, os.path.splitext(os.path.basename(p))[0] + '.txt') for p in image_paths]

images = [load_image(p) for p in image_paths]
labels = [load_labels_yolo(lp, img.shape[1], img.shape[0]) for lp, img in zip(label_paths, images)]

for i in range(0, len(images), 4):
    batch_images = images[i:i+4]
    batch_labels = labels[i:i+4]

    # Ensure there are exactly 4 images for the mosaic
    if len(batch_images) != 4:
        print(f"Skipping batch {i//4}: Not enough images for a mosaic.")
        continue

    mosaic_image, mosaic_labels = create_mosaic(batch_images, batch_labels)
    normalized_labels = normalize_boxes(mosaic_labels, mosaic_image.shape[1], mosaic_image.shape[0])
    save_image(mosaic_image, output_folder, i // 4)
    save_labels(normalized_labels, output_folder, i // 4)

