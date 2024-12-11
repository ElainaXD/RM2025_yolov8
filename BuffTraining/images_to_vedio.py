import shutil

import cv2
import os
from moviepy import ImageSequenceClip

class ImageVideoConverter:
    def __init__(self, img_dir=None, video_path=None, fps=24):
        self.img_dir = img_dir
        self.video_path = video_path
        self.fps = fps
        self.img_files = self._get_sorted_image_files() if img_dir else None


    def _get_sorted_image_files(self):
        img_files = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        img_files.sort(key=self.natural_keys)
        return img_files

    def natural_keys(self, text):
        number_part = os.path.splitext(text)[0].split('_')[-1]
        return int(number_part)

    def create_video(self, output_dir=None, codec='libx264'):
        if not self.img_files:
            print("No image files to convert.")
            return
        clip = ImageSequenceClip(self.img_files, fps=self.fps)
        if output_dir is None:
            output_dir = os.path.dirname(self.img_dir)
        output_name = os.path.basename(self.img_dir) + '_video.mp4'
        output_path = os.path.join(output_dir, output_name)
        clip.write_videofile(output_path, codec=codec)
        print(f"Video saved at {output_path}")

    def convert_video_to_images(self, images_folder=None,interval=30):
        if not self.video_path:
            print("No video path provided.")
            return

        if images_folder is None:
            images_folder_name,_= os.path.splitext(os.path.basename(self.video_path))
            images_folder = os.path.join(os.path.dirname(self.video_path), images_folder_name)

        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count%interval == 0:
                image_name = f"{images_folder}/frame_{(frame_count//interval):04d}.jpg"
                cv2.imwrite(image_name, frame)
                if (frame_count//interval)%10==0 and frame_count//interval>9:
                    print("已输出图片", frame_count//interval)

        cap.release()
        print(f"视频已转换为{frame_count//interval}张图片，并保存在{images_folder}目录下。")

    def renameImgs(self,imgs_folder_path_list,target_folder=None ,save_name='3SE'):
        if target_folder is None:
            print("No target folder provided.")
            return

        # 确保目标文件夹存在，如果不存在则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        for imgs_folder_path in imgs_folder_path_list:
            print(f"Folder:\"{imgs_folder_path}\" start processing")
            for filename in os.listdir(imgs_folder_path):
                # 获取文件的完整路径
                file_path = os.path.join(imgs_folder_path, filename)

                # 检查这是一个文件并且是一个图片文件
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    # 现在使用这个变量来创建新的文件名
                    new_filename = f"{save_name}_{len(os.listdir(target_folder)) + 1}{os.path.splitext(filename)[1]}"

                    # 创建新的文件路径
                    new_file_path = os.path.join(target_folder, new_filename)

                    # 复制并重命名文件
                    shutil.copy2(file_path, new_file_path)
                    print(f"Copied and renamed '{filename}' to '{new_filename}'")
            print(f"Folder:\"{imgs_folder_path}\" has been processed")

        print("All images have been copied and renamed.")
        print(f"total images:{len(os.listdir(target_folder))}")


    # 用于重命名数据集，按照数量命名
    def renameDataset(self,imgs_folder_path,labels_folder_path,target_folder=None ,save_name='3SE'):
        if target_folder is None:
            print("No target folder provided.")
            return

        if imgs_folder_path is None:
            print("No imgs folder provided.")
            return

        if labels_folder_path is None:
            print("No labels folder provided.")
            return

        # 确保目标文件夹存在，如果不存在则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 在target_folder中创建images和labels两个文件夹
        images_folder = os.path.join(target_folder, 'images')
        labels_folder = os.path.join(target_folder, 'labels')

        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)

        # 初始化计数器
        counter = 1

        # 遍历imgs_folder_path中的图片
        for img_file in os.listdir(imgs_folder_path):
            # 获取图片文件名（不包括扩展名）
            img_name, img_ext = os.path.splitext(img_file)

            # 构建对应的标签文件名
            label_file = img_name + '.txt'

            # 检查标签文件是否存在
            if os.path.exists(os.path.join(labels_folder_path, label_file)):
                # 构建新的文件名
                new_img_name = f"{save_name}_{counter}{img_ext}"
                new_label_name = f"{save_name}_{counter}.txt"

                # 构建源文件和目标文件的完整路径
                src_img_path = os.path.join(imgs_folder_path, img_file)
                dst_img_path = os.path.join(images_folder, new_img_name)
                src_label_path = os.path.join(labels_folder_path, label_file)
                dst_label_path = os.path.join(labels_folder, new_label_name)

                # 复制并重命名图片和标签文件
                shutil.copy(src_img_path, dst_img_path)
                shutil.copy(src_label_path, dst_label_path)

                # 更新计数器
                counter += 1
            else:
                print(f"Label file not found for image: {img_file}")

        print("Renaming and copying completed.")


    def get_subfolders(self, folder_path):
        """
        获取指定文件夹下的所有子文件夹，并将它们的绝对路径保存在一个列表中返回。
        参数:
        folder_path (str): 要遍历的文件夹路径。
        返回:
        list: 包含所有子文件夹绝对路径的列表。
        """
        if not os.path.exists(folder_path):
            print("Folder path does not exist.")
            return []

        # 获取该文件夹下的所有文件和文件夹
        entries = os.listdir(folder_path)
        # 过滤出所有文件夹并获取它们的绝对路径
        subfolders = [os.path.abspath(os.path.join(folder_path, entry)) for entry in entries if
                      os.path.isdir(os.path.join(folder_path, entry))]
        return subfolders


if __name__ == '__main__':

    # # 使用示例
    # # 转换视频到图片
    # video_list=[r"D:\BuffDetect\ToDoDataset\BLB3.mp4",r"D:\BuffDetect\ToDoDataset\BLB2.mp4"]
    # for video_path in video_list:
    #     converter = ImageVideoConverter(video_path=video_path)
    #     converter.convert_video_to_images(interval=15)

    # # 转换图片到视频
    # converter = ImageToVideoConverter(img_dir='output_images', fps=40)
    # converter.create_video()

    #将文件夹的图片重新命名并复制到目标文件夹
    converter = ImageVideoConverter()
    # images_folder_list=converter.get_subfolders(r"D:\BuffDetect\ToDoDataset\3SENewDataset")
    # converter.renameImgs(images_folder_list,target_folder=r"D:\BuffDetect\ToDoDataset\resorted")
    converter.renameDataset(r"D:\BuffDetect\3SEDataset\Dataset\野生的\images",
                            r"D:\BuffDetect\3SEDataset\Dataset\野生的\labels2d",
                            r"D:\BuffDetect\3SEDataset\Dataset\野生的\renamed",
                            save_name='wild')