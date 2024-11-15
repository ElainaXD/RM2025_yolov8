import os

def is_valid_line(line):
    # Check if the line starts with '0 ' or '7 '
    class_id = line.split()[0] if line.split() else ''
    return class_id in ['0', '7']

def file_contains_only_class_id_0_and_7(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if not is_valid_line(line.strip()):
                return False
    return True

def find_files_with_only_class_id_0_and_7(root_folder, output_file):
    with open(output_file, 'w') as out_file:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.txt'):
                    full_path = os.path.join(root, file)
                    if file_contains_only_class_id_0_and_7(full_path):
                        out_file.write(full_path + '\n')
                        print(full_path)

root_folder = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data\labels\train"  # Replace with the path to your root folder
output_file = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\output1.txt"   # Replace with the path to your desired output file

find_files_with_only_class_id_0_and_7(root_folder, output_file)
print(f"Processed files are listed in: {output_file}")
