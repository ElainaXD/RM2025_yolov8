from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import shutil


def split_train_val(image_source_path, label_source_path, target_path):
    if not Path(image_source_path).is_dir():
        print("Path does not exist!")
        return False

    if not Path(label_source_path).is_dir():
        print("Path does not exist!")
        return False

    if not Path(target_path).is_dir():
        os.makedirs(Path(target_path), exist_ok=True)

    total_files = []
    for filename in os.listdir(image_source_path):
        total_files.append(filename)

    # test_size为训练集和测试集的比例
    # random_state为none时，每次划分都是随机
    train_files, val_files = train_test_split(total_files, test_size=0.25, random_state=42)
    train_set_images_dir = os.path.join(target_path, "images", "train")
    val_set_images_dir = os.path.join(target_path, "images", "val")
    train_set_labels_dir = os.path.join(target_path, "labels", "train")
    val_set_labels_dir = os.path.join(target_path, "labels", "val")

    if not Path(train_set_images_dir).is_dir() or not Path(val_set_images_dir).is_dir():
        os.makedirs(train_set_images_dir)
        os.makedirs(val_set_images_dir)
        os.makedirs(train_set_labels_dir)
        os.makedirs(val_set_labels_dir)

    for i in range(len(train_files)):
        train_images_path = os.path.join(image_source_path, train_files[i])
        label_file=os.path.splitext(train_files[i])[0]+".txt"
        train_labels_path = os.path.join(label_source_path, label_file)
        shutil.copy(train_images_path, train_set_images_dir)
        shutil.copy(train_labels_path, train_set_labels_dir)

    for j in range(len(val_files)):
        val_images_path = os.path.join(image_source_path, val_files[j])
        label_file=os.path.splitext(val_files[j])[0]+".txt"
        val_labels_path = os.path.join(label_source_path, label_file)
        shutil.copy(val_images_path, val_set_images_dir)
        shutil.copy(val_labels_path, val_set_labels_dir)


if __name__ == '__main__':
    source_images_path = r'D:\BuffDetect\2023年西交利物浦GMaster战队部分内录和所有数据集\能量机关数据集\XJTLU_2023_WIN_ALL\images'  # 图片路径
    source_labels_path = r'D:\BuffDetect\2023年西交利物浦GMaster战队部分内录和所有数据集\能量机关数据集\XJTLU_2023_WIN_ALL\labels'
    target_path = r'D:\Github\RM2025_yolov8\BuffTraining\BuffDataset'  # 划分测试集存放路径
    split_train_val(source_images_path, source_labels_path, target_path)
    print("划分完成！")
