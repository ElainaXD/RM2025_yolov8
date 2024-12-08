from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np

class Split:
    def __init__(self, img_source_dir,label_source_dir):
        self.img_source_dir = img_source_dir
        self.label_source_dir = label_source_dir

    # 用于分割数据集与训练集
    def split_train_val(self,target_path):
        if not Path(self.img_source_dir).is_dir():
            print("Path does not exist!")
            return False

        if not Path(self.label_source_dir).is_dir():
            print("Path does not exist!")
            return False

        if not Path(target_path).is_dir():
            os.makedirs(Path(target_path), exist_ok=True)

        total_files = []
        for filename in os.listdir(self.img_source_dir):
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
            train_images_path = os.path.join(self.img_source_dir, train_files[i])
            label_file=os.path.splitext(train_files[i])[0]+".txt"
            train_labels_path = os.path.join( self.label_source_dir, label_file)
            shutil.copy(train_images_path, train_set_images_dir)
            shutil.copy(train_labels_path, train_set_labels_dir)

        for j in range(len(val_files)):
            val_images_path = os.path.join(self.img_source_dir, val_files[j])
            label_file=os.path.splitext(val_files[j])[0]+".txt"
            val_labels_path = os.path.join( self.label_source_dir, label_file)
            shutil.copy(val_images_path, val_set_images_dir)
            shutil.copy(val_labels_path, val_set_labels_dir)

    # 用于分拣目标标签，返回值为两个列表，一个是问题标签列表，这一部分的存在标签格式错误，另一部分是目标标签列表
    # 两个列表中包含的是文件名，包括后缀txt
    def sort_target_labels(self,target_labels):
        problem_labels_list = []
        target_labels_list = []
        rest_labels_list = []  # 新增列表存储剩余的标签文件名

        # 遍历文件夹中的所有文件
        for filename in os.listdir(self.label_source_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.label_source_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    numbers = np.array(content.split(), dtype=float)

                    # 检查数字的个数是否是15的倍数
                    if len(numbers) % 15 != 0:
                        problem_labels_list.append(filename)
                        continue  # 跳过当前文件，继续下一个文件

                    # 检查每15个数的开头数
                    is_target_or_problem = False  # 标记是否是目标或问题标签
                    for i in range(0, len(numbers), 15):
                        group = numbers[i:i + 15]
                        if group[0] not in [0, 1, 2, 3]:
                            problem_labels_list.append(filename)
                            is_target_or_problem = True
                            break  # 跳出循环，继续下一个文件
                        elif group[0] in target_labels:
                            target_labels_list.append(filename)
                            is_target_or_problem = True
                            break  # 跳出循环，继续下一个文件

                    # 如果既不是目标标签也不是问题标签，则添加到剩余列表
                    if not is_target_or_problem:
                        rest_labels_list.append(filename)

        return problem_labels_list, target_labels_list, rest_labels_list

    # 按照目标分割图像与标签并将其放到目标文件夹中
    def split_target(self, target_folder_path, target_labels):
        problem_labels_list, target_labels_list, rest_labels_list = self.sort_target_labels(target_labels)

        # 创建目标文件夹路径下的子文件夹
        problem_labels_dir = os.path.join(target_folder_path, 'problem', 'labels')
        problem_images_dir = os.path.join(target_folder_path, 'problem', 'images')
        target_labels_dir = os.path.join(target_folder_path, 'target', 'labels')
        target_images_dir = os.path.join(target_folder_path, 'target', 'images')
        rest_labels_dir = os.path.join(target_folder_path, 'rest', 'labels')
        rest_images_dir = os.path.join(target_folder_path, 'rest', 'images')

        os.makedirs(problem_labels_dir, exist_ok=True)
        os.makedirs(problem_images_dir, exist_ok=True)
        os.makedirs(target_labels_dir, exist_ok=True)
        os.makedirs(target_images_dir, exist_ok=True)
        os.makedirs(rest_labels_dir, exist_ok=True)  # 创建rest文件夹
        os.makedirs(rest_images_dir, exist_ok=True)  # 创建rest文件夹

        # 定义一个函数来复制文件
        def copy_files(file_list, label_dir, image_dir):
            for filename in file_list:
                label_filename = filename
                image_filename = filename.rsplit('.', 1)[0] + '.jpg'  # 假设图片文件扩展名为.jpg
                label_source_path = os.path.join(self.label_source_dir, label_filename)
                image_source_path = os.path.join(self.img_source_dir, image_filename)
                shutil.copy(label_source_path, label_dir)
                if os.path.exists(image_source_path):
                    shutil.copy(image_source_path, image_dir)

        # 复制problem_labels_list中的文件到problem文件夹
        copy_files(problem_labels_list, problem_labels_dir, problem_images_dir)

        # 复制target_labels_list中的文件到target文件夹
        copy_files(target_labels_list, target_labels_dir, target_images_dir)

        # 复制rest_labels_list中的文件到rest文件夹
        copy_files(rest_labels_list, rest_labels_dir, rest_images_dir)

if __name__ == '__main__':
    source_images_path = r"D:\BuffDetect\3SEDataset\Version1\images"  # 图片路径
    source_labels_path = r"D:\BuffDetect\3SEDataset\Version1\labels"
    target_path = r"D:\BuffDetect\3SEDataset\Version1_splited" # 划分测试集存放路径
    split=Split(source_images_path,source_labels_path)
    # split.split_target(target_folder_path=target_path,target_labels=[0,2])
    split.split_train_val(target_path)
    print("划分完成！")
