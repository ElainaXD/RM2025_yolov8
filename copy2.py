import os
import shutil

def copy_folders(txt_source, jpg_source, destination):
    # Walk through the txt source directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(txt_source):
        # Determine the corresponding jpg source directory
        jpg_dirpath = dirpath.replace(txt_source, jpg_source)

        # Create a corresponding directory in the destination
        dest_dir = os.path.join(destination, os.path.relpath(dirpath, txt_source))
        os.makedirs(dest_dir, exist_ok=True)

        # Check for each .txt file if corresponding .jpg file exists in the jpg source
        for filename in filenames:
            if filename.lower().endswith('.txt'):  # Check if the file is a .txt
                base_name = os.path.splitext(filename)[0]
                jpg_file = base_name + '.jpg'
                jpg_file_path = os.path.join(jpg_dirpath, jpg_file)

                # Check if the .jpg file exists
                if os.path.exists(jpg_file_path):
                    dest_file = os.path.join(dest_dir, jpg_file)
                    shutil.copy2(jpg_file_path, dest_file)
                    print(f"Copied: {jpg_file}")
                else:
                    print(f"No corresponding jpg found for {filename}")

# Usage example

jpg_source_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data\images\train"
txt_source_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v3\labels\train"
destination_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v3\images\train"

copy_folders(txt_source_folder, jpg_source_folder, destination_folder)
