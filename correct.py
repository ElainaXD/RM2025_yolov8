import os
import asyncio
import aiofiles
from tqdm import tqdm

async def check_file(file_path, semaphore, found_files):
    async with semaphore:
        async with aiofiles.open(file_path, 'r') as file:
            async for line in file:
                if '0.000000' in line:
                    found_files.append(file_path)
                    break

async def find_files_with_zero_class_id(root_dir, output_file, batch_size=200):
    txt_files = [os.path.join(root, file) for root, dirs, files in os.walk(root_dir) for file in files if file.endswith('.txt')]
    progress = tqdm(total=len(txt_files), desc="Processing Files")
    semaphore = asyncio.Semaphore(100)
    found_files = []

    for i in range(0, len(txt_files), batch_size):
        batch = txt_files[i:i + batch_size]
        tasks = [asyncio.create_task(check_file(file_path, semaphore, found_files)) for file_path in batch]
        await asyncio.gather(*tasks)
        progress.update(len(batch))

    progress.close()

    # Write all found file paths to the output file
    async with aiofiles.open(output_file, 'a') as out:
        for file_path in found_files:
            await out.write(f"{file_path}\n")

root_directory = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\data_v2\labels"
output_file = r"\\192.168.77.100\Model_Center\Yolo\YoloV8CY\yolov8pose\float.txt"
asyncio.run(find_files_with_zero_class_id(root_directory, output_file))

#7239