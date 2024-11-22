from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import shutil

from tensorflow.python.autograph import convert


class Convert:
    def __init__(self,label_fold_path):
        if not Path(label_fold_path).is_dir():
            print("Path does not exist!")
        else:
            self.label_fold_path = label_fold_path
    #用来给roboflow用的，但是寄了，因为要求所有的关键点都在框内，但西埔的数据集是在框外的
    # pose有两种数据集格式，一种是二维，一种是三维，三维比二维多了一个表示该点是否可见的数据，1表示可见，2表示不可见，0表示未标注
    def pose_2d_to_3d(self,res_fold_path):
        if not Path(res_fold_path).is_dir():
            os.makedirs(Path(res_fold_path), exist_ok=True)
        for filename in os.listdir(self.label_fold_path):
            file_path = os.path.join(self.label_fold_path, filename)
            with open(file_path,'r') as f:

                line = f.readlines()[0]
                numbers=line.split()
                new_data = numbers[:5]
                # 从第五号数字开始，每添加两个数，再添加一个1
                for i in range(5, len(numbers), 2):
                    new_data.append(numbers[i])
                    new_data.append(numbers[i + 1])
                    new_data.append('1')
                output_flie_path=os.path.join(res_fold_path, filename)
                # 使用with语句打开文件，确保文件操作完成后关闭文件
                with open(output_flie_path, 'w') as file:
                    # 遍历数据列表，并将每个数据项写入文件，每个数据项后跟一个空格
                    for data in new_data:
                        file.write(data + ' ')


if __name__ == '__main__':
    data_convert=Convert(r"D:\BuffDetect\testOriginNum")
    data_convert.pose_2d_to_3d(res_fold_path=r"D:\BuffDetect\testOutputNum")


