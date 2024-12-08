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

    # pose有两种数据集格式，一种是二维，一种是三维，三维比二维多了一个表示该点是否可见的数据，1表示可见，2表示不可见，0表示未标注
    # 0是2d到3d，1是3d到2d
    # conf==1时3D转2D，conf==0时2D转3D
    def pose_2d_3d(self,res_fold_path,conf):
        if conf not in [0,1]:
            print("请选择正确的conf参数\n0:2D->3D\n1:3D->2D")
        if not Path(res_fold_path).is_dir():
            os.makedirs(Path(res_fold_path), exist_ok=True)
        for filename in os.listdir(self.label_fold_path):
            file_path = os.path.join(self.label_fold_path, filename)
            with open(file_path,'r') as f:
                lines = f.readlines()
                res_data=[]
                for line in lines:
                    numbers=line.split()
                    new_data = numbers[:5]
                    if conf == 0:
                        # 从第五号数字开始，每添加两个数，再添加一个1
                        for i in range(5, len(numbers), 2):
                            new_data.append(numbers[i])
                            new_data.append(numbers[i + 1])
                            new_data.append('1')
                    else:
                        # 删除3D姿态末尾的数，也就是不添加
                        for i in range(5, len(numbers), 3):
                            new_data.append(numbers[i])
                            new_data.append(numbers[i + 1])
                    for i in new_data:
                        res_data.append(i)
                output_flie_path=os.path.join(res_fold_path, filename)
                # 使用with语句打开文件，确保文件操作完成后关闭文件
                with open(output_flie_path, 'w') as file:
                    if conf == 0:
                        # 遍历数据列表，并将每个数据项写入文件，每个数据项后跟一个空格
                        for i, number in enumerate(res_data, 1):
                            # 添加数字和空格（如果不是每组的最后一个数字）
                            file.write(f"{number} ")
                            # 每十五个数字后添加换行符
                            if i % 20 == 0:
                                file.write("\n")
                    else:
                        # 遍历数据列表，并将每个数据项写入文件，每个数据项后跟一个空格
                        for i, number in enumerate(res_data, 1):
                            # 添加数字和空格（如果不是每组的最后一个数字）
                            file.write(f"{number} ")
                            # 每十五个数字后添加换行符
                            if i % 15 == 0:
                                file.write("\n")


if __name__ == '__main__':
    data_convert=Convert(r"D:\BuffDetect\3SEDataset\BuffPose\labels")
    data_convert.pose_2d_3d(res_fold_path=r"D:\BuffDetect\3SEDataset\BuffPose\labels2d",conf=1)


