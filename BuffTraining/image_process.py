import os
import shutil
import cv2
import numpy as np
import math
from rich.progress import Progress

class ImageProcess:
    def __init__(self):
        self.key_point_color = [[255, 0, 255], [255, 0, 255], [0, 215, 255], [255, 0, 255], [255, 0, 255]]
        self.mask_color=[0,0,0]


    # 用于得到一个图片的结果，返回一个list：x,y,w,h和剩下的点（x_i,y_i）
    def get_pixel_label(self, image_path, label_path):
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape

        with open(label_path, 'r') as f:
            content = f.read()
            data=np.array(content.split(),dtype=float)
            if len(data)%15!=0:
                print("Not a available txt file!")
                return

            pixel_result=[]

            for i in range(0,len(data),15):
                group=data[i:i+15]
                if group[0] not in [0,1,2,3]:
                    print("Not a available txt file!")
                    return
                else:
                    target_result = []
                    cls=group[0]
                    x,y,w,h=group[1:5]
                    x_pixel = int((x * img_width) - (w * img_width / 2))
                    y_pixel = int((y * img_height) - (h * img_height / 2))
                    width_pixel = int(w * img_width)
                    height_pixel = int(h * img_height)

                    for number in [cls,x_pixel,y_pixel,width_pixel,height_pixel]:
                        target_result.append(number)
                    for j in range(5, len(group), 2):
                        point_x=int(group[j]* img_width)
                        point_y=int(group[j+1]* img_height)
                        target_result.append(point_x)
                        target_result.append(point_y)
                    pixel_result.append(target_result)
        return pixel_result

    def show_image_label(self, image_path, label_path):
        image = cv2.imread(image_path)
        label_list=self.get_pixel_label(image_path, label_path)
        for label in label_list:
            cls=label[0]
            x_pixel,y_pixel,width_pixel,height_pixel=label[1:5]
            label_place = (x_pixel, y_pixel - 10)
            cv2.rectangle(image, (x_pixel, y_pixel), (x_pixel + width_pixel, y_pixel + height_pixel),
                          (0, 255, 0), 2)
            cv2.putText(
                img=image,
                text=str(cls),
                org=label_place,
                color=(0, 255, 0),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1.0,  # 添加 fontScale 参数，这里假设使用 1.0 作为字体大小
                thickness=2,
                lineType=cv2.LINE_AA
            )
            if cls == 0 or cls == 2:
                for j in range(5,len(label),2):
                    point_x=label[j]
                    point_y=label[j+1]
                    cv2.putText(
                        img=image,
                        text=str((j-5)//2),
                        org=(point_x-10,point_y-10),
                        color=(0, 255, 0),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.0,  # 添加 fontScale 参数，这里假设使用 1.0 作为字体大小
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                    cv2.circle(image,(point_x,point_y),4,self.key_point_color[(j-5)//2],-1)

        cv2.imshow('Image with Rectangle and Label', image)
        cv2.waitKey(0)

    def mask_peripheral_light(self, image_path, label_path):
        image = cv2.imread(image_path)
        # 获取图像尺寸
        height, width = image.shape[:2]
        # 创建二值化图像，用来绘制圆环
        binary_image = np.zeros((height, width), dtype=np.uint8)


        label_list=self.get_pixel_label(image_path, label_path)
        for label in label_list:
            cls=label[0]
            if cls == 0 or cls == 2:
                x_pixel,y_pixel,width_pixel,height_pixel=label[1:5]
                keypoint=[]
                x_sum=0
                y_sum=0
                for i in range(5, len(label), 2):
                    point_x = label[i]
                    point_y = label[i + 1]
                    keypoint.append((point_x,point_y))
                    if i != 13:
                        x_sum += point_x
                        y_sum += point_y



                diameter=(np.linalg.norm(np.array(keypoint[0]) - np.array(keypoint[3]))
                          +np.linalg.norm(np.array(keypoint[1]) - np.array(keypoint[4])))/2
                delta_radius=(math.sqrt(width_pixel**2+height_pixel**2)-diameter)/2
                inner_radius=math.floor(diameter/2*0.68)
                outer_radius=math.ceil((delta_radius+inner_radius)*1.2)
                # print(inner_radius, outer_radius)
                R_center=(label[13],label[14])
                center=(round(x_sum/4),round(y_sum/4))

                # 这里发现单纯划线不大理想，因为线的两段是圆的，所以再画垂直于此处的线来抹平
                origin_vector=(R_center[0]-center[0],R_center[1]-center[1])
                vertical_vector=np.array([-origin_vector[1],origin_vector[0]])
                norm_vertical_vector=vertical_vector/np.linalg.norm(vertical_vector)
                # 定义垂直线的长度
                line_length = diameter

                # 绘制一条直线来去除中间这一道
                start_point = (round(center[0] + (R_center[0] - center[0]) / 2.8), round(center[1] + (R_center[1] - center[1]) / 2.8))

                # 计算垂直线的两个端点
                vertical_start_point = start_point - norm_vertical_vector * line_length / 2
                vertical_end_point = start_point + norm_vertical_vector * line_length / 2
                vertical_start_point=tuple(vertical_start_point.astype(int))
                vertical_end_point=tuple(vertical_end_point.astype(int))
                if not (isinstance(center, tuple) and len(center) == 2):
                    return None
                if not (isinstance(outer_radius, int) or isinstance(outer_radius, float)) or outer_radius < 0:
                    return None
                # 绘制圆环
                cv2.circle(binary_image, center, outer_radius,  (255), -1)
                cv2.circle(binary_image, center, inner_radius, (0), -1)
                try:
                    vertical_thickness=max(1,round(inner_radius*0.6))
                    cv2.line(binary_image, vertical_start_point, vertical_end_point, (0), vertical_thickness,lineType=cv2.LINE_8)
                    cv2.line(binary_image, start_point, R_center, (0), vertical_thickness,lineType=cv2.LINE_8)

                except cv2.error as e:
                    print('发生 OpenCV 错误:', e)
                    print(f"image path: {image_path}")
                    return None
        # 遍历这个二值化图像的所有点，如果是白色那就把同样位置的区域在原图上涂成黑色
        for y in range(height):
            for x in range(width):
                if binary_image[y][x] == 255:
                    image[y][x] = [0,0,0]
        return image

    def mask_images(self, image_path, label_path, result_folder_path):
        origin_images_dir=os.path.join(result_folder_path,'origin_images')
        origin_labels_dir=os.path.join(result_folder_path,'origin_labels')
        mask_images_dir=os.path.join(result_folder_path,'mask_images')
        mask_labels_dir=os.path.join(result_folder_path,'mask_labels')
        error_images_dir=os.path.join(result_folder_path,'error_images')
        error_label_dir=os.path.join(result_folder_path,'error_labels')

        os.makedirs(origin_images_dir, exist_ok=True)
        os.makedirs(origin_labels_dir, exist_ok=True)
        os.makedirs(mask_images_dir, exist_ok=True)
        os.makedirs(mask_labels_dir, exist_ok=True)
        os.makedirs(error_images_dir, exist_ok=True)
        os.makedirs(error_label_dir, exist_ok=True)

        # 获取文件夹下所有文件和文件夹的名称列表
        img_list = os.listdir(image_path)
        # 计算文件的数量
        file_count = sum(1 for entry in img_list if os.path.isfile(os.path.join(image_path, entry)))

        with Progress() as progress:
            task = progress.add_task("Processing...", total=file_count)

            for filename in img_list:
                if filename.endswith('.jpg'):
                    label_name,_=os.path.splitext(filename)
                    label_name=f"{label_name}.txt"

                    img_dir = os.path.join(image_path, filename)
                    label_dir=os.path.join(label_path,label_name)

                    origin_image_new_name=f"origin_{len(os.listdir(origin_images_dir))+1}.jpg"
                    mask_image_new_name=f"mask_{len(os.listdir(mask_images_dir))+1}.jpg"
                    origin_label_new_name=f"origin_{len(os.listdir(origin_labels_dir))+1}.txt"
                    mask_label_new_name=f"mask_{len(os.listdir(mask_labels_dir))+1}.txt"

                    origin_image_new_path=os.path.join(origin_images_dir, origin_image_new_name)
                    mask_image_new_path=os.path.join(mask_images_dir, mask_image_new_name)
                    origin_label_new_path=os.path.join(origin_labels_dir, origin_label_new_name)
                    mask_label_new_path=os.path.join(mask_labels_dir, mask_label_new_name)

                    mask_image = self.mask_peripheral_light(img_dir, label_dir)
                    if mask_image is None:
                        error_image_new_name = f"error_{len(os.listdir(error_images_dir)) + 1}.jpg"
                        error_image_new_path = os.path.join(error_images_dir, error_image_new_name)
                        error_label_new_name = f"error_label_{len(os.listdir(error_label_dir))+1}.txt"
                        error_label_new_path = os.path.join(error_label_dir, error_label_new_name)
                        print(f"图像出错，图像地址为{img_dir},将该图像保存到{error_image_new_path}")
                        shutil.copy(img_dir, error_image_new_path)
                        shutil.copy(label_dir, error_label_new_path)
                    else:
                        shutil.copy(img_dir, origin_image_new_path)
                        shutil.copy(label_dir, origin_label_new_path)
                        shutil.copy(label_dir, mask_label_new_path)
                        cv2.imwrite(mask_image_new_path, mask_image)
                progress.update(task, advance=1)



if __name__ == "__main__":
    # image_path=r"D:\BuffDetect\3SEDataset\BuffPose\renamed\images\3SE_1203.jpg"
    # label_path=r"D:\BuffDetect\3SEDataset\BuffPose\renamed\labels\3SE_1203.txt"
    # image_process=ImageProcess()
    # image_process.mask_peripheral_light(image_path,label_path)
    # cv2.imshow("test", image_process.mask_peripheral_light(image_path,label_path))
    # cv2.waitKey(0)
    #
    imgs_path=r"D:\BuffDetect\3SEDataset\BuffPose\renamed\images"
    labels_path=r"D:\BuffDetect\3SEDataset\BuffPose\renamed\labels"
    image_process=ImageProcess()
    image_process.mask_images(imgs_path, labels_path, result_folder_path=r"D:\BuffDetect\3SEDataset\BuffPose\renamed\masked")
    # image_process.show_image_label(r"C:\Users\wyhao\Desktop\train\images\dc23abb9-3431_jpg.rf.b101a0506df3602ddf894e0ce08adab5.jpg",r"C:\Users\wyhao\Desktop\train\dc23abb9-3431_jpg.rf.b101a0506df3602ddf894e0ce08adab5.txt")