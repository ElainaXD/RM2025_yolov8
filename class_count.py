import os
import asyncio
import aiofiles
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt

async def process_file(file_path, class_counts, progress):
    async with aiofiles.open(file_path, 'r') as file:
        async for line in file:
            class_id = line.split()[0]
            if class_id in class_counts:
                class_counts[class_id] += 1
            else:
                class_counts[class_id] = 1
    progress.update(1)  # Update the progress bar after each file

async def count_classes(root_dir):
    class_counts = {}
    files_to_process = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)  # Define file_path here
                print(f"Found file: {file_path}")  # Now file_path is defined
                files_to_process.append(file_path)

    # Initialize tqdm progress bar
    progress = tqdm(total=len(files_to_process), desc="Processing Files")

    while files_to_process:
        batch = files_to_process[:200]
        files_to_process = files_to_process[200:]
        await asyncio.gather(*(process_file(file, class_counts, progress) for file in batch))
    
    progress.close()  # Close the progress bar

    return class_counts


def plot_and_save_barchart(class_counts, output_path):
    if not class_counts:
        print("No class data found to plot.")
        return

    # Sorting the class counts
    sorted_classes = sorted(class_counts.items())
    classes, counts = zip(*sorted_classes)

    # Creating the bar chart
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Class Counts for YOLO Training')

    # Save the plot
    plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory


async def main():
    root_directory = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data\labels"
    output_file = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\bar_chart1.png"
    print("Starting to count classes...")
    class_counts = await count_classes(root_directory)
    print("Finished counting. Plotting bar chart...")
    plot_and_save_barchart(class_counts, output_file)
    print(f"Bar chart saved to {output_file}")

asyncio.run(main())
