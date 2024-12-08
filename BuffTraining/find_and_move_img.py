import os
import shutil

def copy_jpg_files(source_folder, destination_folder, txt_file):
    # 读取txt文件中的jpg文件名
    with open(txt_file, 'r') as file:
        jpg_filenames = file.read().splitlines()

    # 遍历文件名列表
    for filename in jpg_filenames:
        # 构建完整的文件路径
        source_file_path = os.path.join(source_folder, filename)
        # 检查文件是否存在
        if os.path.isfile(source_file_path):
            # 构建目标文件夹路径
            destination_file_path = os.path.join(destination_folder, filename)
            # 复制文件
            shutil.copy2(source_file_path, destination_file_path)
            print(f"文件 '{filename}' 已复制到 '{destination_folder}'.")
        else:
            print(f"文件 '{filename}' 在源文件夹中未找到。")

# 使用示例
source_folder = r'D:\BuffDetect\2023年西交利物浦GMaster战队部分内录和所有数据集\能量机关数据集\XJTLU_2023_WIN_ALL\images'  # 源文件夹路径
destination_folder = r"D:\BuffDetect\WebJpg\batch2"  # 目标文件夹路径
txt_file = r"D:\BuffDetect\WebJpg\batch2.txt"  # 包含jpg文件名的文本文件路径

# 确保目标文件夹存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 执行复制操作
copy_jpg_files(source_folder, destination_folder, txt_file)
