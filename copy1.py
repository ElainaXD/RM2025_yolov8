import os
import shutil

def copy_folders(source, destination, exclude_file):
    # Read the list of files to exclude
    with open(exclude_file, 'r') as file:
        excluded_files = {line.strip() for line in file}

    # Walk through the source directory
    for dirpath, dirnames, filenames in os.walk(source):
        # Create a corresponding directory in the destination
        dest_dir = os.path.join(destination, os.path.relpath(dirpath, source))
        os.makedirs(dest_dir, exist_ok=True)

        # Copy files, excluding the ones in the list
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)

            # Check if the full path of the file is in the excluded list
            if src_file not in excluded_files:
                dest_file = os.path.join(dest_dir, filename)
                shutil.copy2(src_file, dest_file)
                print(f"Copied: {filename}")
            else:
                print(f"Excluded: {filename}")

# Usage example
# source_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data\labels\val"
# destination_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v3\labels\val"
# exclude_file = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\output.txt"

source_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data\labels\train"
destination_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v3\labels\train"
exclude_file = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\output1.txt"

copy_folders(source_folder, destination_folder, exclude_file)

