import os

def correct_lingering_zeros(value):
    try:
        if float(value) == int(float(value)):
            return str(int(float(value)))
    except ValueError:
        pass
    return value

def correct_file_annotations(input_file, output_folder):
    base_filename = os.path.basename(input_file)
    output_file = os.path.join(output_folder, base_filename)

    with open(input_file, 'r') as file:
        lines = file.readlines()

    corrected_annotations = []
    corrections_made = False
    for line in lines:
        values = line.strip().split()
        corrected_values = [correct_lingering_zeros(value) for value in values]
        if corrected_values != values:
            corrections_made = True
        corrected_annotations.append(' '.join(corrected_values))

    if corrections_made:
        with open(output_file, 'w') as file:
            for line in corrected_annotations:
                file.write(f"{line}\n")
        print(f"Corrected file saved: {output_file}")
    else:
        print(f"No corrections needed for: {input_file}")

def correct_annotations_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)
            correct_file_annotations(input_file, output_folder)

# Specify the input and output folder paths
input_folder_path = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v2\labels\train"  # Replace with the path to your input folder
output_folder_path = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\corrected1"  # Replace with the path to your output folder

# Run the correction process
correct_annotations_in_folder(input_folder_path, output_folder_path)
