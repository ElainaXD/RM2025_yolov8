import math
import os
import cv2
import torch
import numpy as np
from tensorboard.compat.tensorflow_stub.dtypes import float32

from ultralytics import YOLO
import numpy as np

from ultralytics.trackers.utils import kalman_filter
from ultralytics.utils.plotting import colors


class ExtractCentre:
    def __init__(self, color='red'):
        """
        初始化函数，设置提取颜色。

        参数:
        color (str): 提取的颜色，'red' 或 'blue'。
        """
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_blue = np.array([100, 150, 0])
        self.upper_blue = np.array([140, 255, 255])
        self.color = color.lower()

    def extract_color(self, image_np):
        """
        提取图像中指定颜色的部分并返回二值化图像。

        参数:
        image_np (numpy.ndarray): 已读取的图像，类型为NumPy数组。

        返回:
        numpy.ndarray: 二值化后的图像。
        """
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        if self.color == 'red':
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif self.color == 'blue':
            mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        else:
            raise ValueError("Unsupported color. Use 'red' or 'blue'.")

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary

    def calculate_center(self, binary_image, center_point, radius):
        """
        在二值化图像的圆形区域内计算所有白色点的中心。

        参数:
        binary_image (numpy.ndarray): 二值化图像。
        center_point (tuple): 圆心点坐标(x, y)。
        radius (int): 圆的半径。

        返回:
        tuple: 计算得到的白色点的中心坐标(x, y)，如果找不到则返回None。
        """
        mask = np.zeros_like(binary_image)
        cv2.circle(mask, center_point, radius, 255, -1)
        masked_image = cv2.bitwise_and(binary_image, mask)
        white_points = np.argwhere(masked_image == 255)

        if len(white_points) == 0:
            return None

        center_x = int(np.mean(white_points[:, 1]))
        center_y = int(np.mean(white_points[:, 0]))

        return (center_x, center_y)

    def get_R_centre(self,image_np,estimate_point, radius):
        binary_image = self.extract_color(image_np)
        point=self.calculate_center(binary_image, estimate_point, radius)
        return point

class RecursiveLeastSquares:
    def __init__(self, num_params, initial_A=None, initial_P=None, delta=0.98):
        """
        初始化RLS算法
        :param num_params: 参数的数量
        :param initial_A: 初始参数估计 (可选)
        :param initial_P: 初始误差协方差矩阵 (可选)
        :param delta: 忘记因子 (可选，默认为0.98)
        """
        self.num_params = num_params
        self.delta = delta

        if initial_A is None:
            self.A_hat = np.zeros((num_params, 1))
        else:
            if initial_A.ndim == 1:
                initial_A = initial_A.reshape(-1, 1)
            elif initial_A.shape[1] != 1:
                raise ValueError("initial_A must be a column vector")
            self.A_hat = initial_A
        if initial_P is None:
            self.P = np.eye(num_params) * 1000  # 初始协方差矩阵，较大的值表示不确定性
        else:
            self.P = initial_P

    def update(self, X, B):
        """
        使用新的数据点更新参数估计
        :param X: 输入数据向量 (列向量形式)
        :param B: 输出数据值
        """
        X = X.reshape(-1, 1)  # 确保X是列向量
        B = B.reshape(-1, 1)  # 确保B是列向量

        # 计算预测误差
        epsilon = B - X.T @ self.A_hat

        # 计算增益矩阵
        S = X.T @ self.P @ X + self.delta
        K = (self.P @ X) / S

        # 更新参数估计
        self.A_hat += K * epsilon

        # 更新误差协方差矩阵
        self.P = (self.P - K @ X.T @ self.P) / self.delta

    def get_params(self):
        """
        获取当前的参数估计
        """
        return self.A_hat


class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2, 0)

        # 初始化状态转移矩阵、观测矩阵、过程噪声协方差矩阵和测量噪声协方差矩阵
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)

        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32) * 0.03

        self.kf.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], dtype=np.float32) * 0.1

        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

    def correct(self, measurement):
        self.kf.predict()
        # 使用correct方法进行滤波
        measurement = np.array(measurement, dtype=np.float32).reshape(-1, 1)
        estimated_state = self.kf.correct(measurement)
        return estimated_state[:2].flatten()  # 返回滤波后的状态(x, y)


class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, control_dim=0):
        self.measurement_dim=measurement_dim
        # 初始化卡尔曼滤波器
        self.kf = cv2.KalmanFilter(state_dim, measurement_dim, control_dim)

        # 初始化状态转移矩阵（假设状态是静态的）
        self.kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)

        # 初始化观测矩阵（假设我们可以直接观测到整个状态）
        self.kf.measurementMatrix = np.eye(measurement_dim, dtype=np.float32)

        # 初始化过程噪声协方差矩阵
        self.kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * 1e-5

        # 初始化测量噪声协方差矩阵
        self.kf.measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * 1e-1

        # 初始化误差协方差矩阵
        self.kf.errorCovPost = np.eye(state_dim, dtype=np.float32)

        # 初始化状态向量
        self.kf.statePost = np.zeros(state_dim, dtype=np.float32)

    # def predict(self):
    #     # 预测下一个状态
    #     return self.kf.predict()

    def correct(self, measurement):
        self.kf.predict()
        measurement = np.array(measurement, dtype=np.float32).reshape(-1, 1)
        estimated_state = self.kf.correct(measurement)
        return estimated_state[:self.measurement_dim].flatten()  # 返回滤波后的状态(x, y)

    def get_state(self):
        # 获取当前状态估计
        return self.kf.statePost

class Predict_pose_visualize:
    def __init__(self, model_path,predict_trace=False):
        self.rls = None
        self.model_path = model_path
        self.model=YOLO(model_path)
        self.key_point_color=[[255,0,255],[255,0,255],[255,0,255],[255,0,255],[0,215,255]]
        self.device=device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model_name=os.path.splitext(os.path.basename(model_path))[0]

    def calculate_ellipse(self, point, img_size):
        division=min(img_size[0], img_size[1])
        norm_point = (point[0] / division, point[1] / division)

        X = np.float32([norm_point[0] ** 2, norm_point[0] * norm_point[1], norm_point[1] ** 2, norm_point[0], norm_point[1], 1])
        self.rls.update(X, np.float32([0]))
        A,B,C,D,E,F = self.rls.get_params().flatten()
        # 计算旋转角度
        theta = 0.5 * math.atan2(B, A - C)

        # 旋转矩阵
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        # 计算旋转后的系数
        A_rot = A * cos_theta ** 2 + B * cos_theta * sin_theta + C * sin_theta ** 2
        C_rot = A * sin_theta ** 2 - B * cos_theta * sin_theta + C * cos_theta ** 2
        D_rot = D * cos_theta + E * sin_theta
        E_rot = -D * sin_theta + E * cos_theta
        F_rot = F

        # 计算椭圆中心
        h_rot = - D_rot/(2*A_rot)
        k_rot = -E_rot/(2*C_rot)

        constant_term = -F_rot + (0.5* D_rot)**2+(0.5*E_rot)**2

        # 计算半轴长度
        a_squared = constant_term / abs(A_rot)
        b_squared = constant_term / abs(C_rot)

        a_rot = math.sqrt(a_squared)
        b_rot = math.sqrt(b_squared)

        # 将归一化的椭圆参数转换回原始坐标系
        h_rot = h_rot * division
        k_rot = k_rot * division
        a_rot = a_rot * division
        b_rot = b_rot * division
        res_para=[h_rot, k_rot, a_rot, b_rot]
        res=[res_para]
        res.append(rotation_matrix)
        return res

    def draw_ellipse(self, para,rotation_matrix,img,centre):
        h_rot, k_rot, a_rot, b_rot = para
        height, width = img.shape[0], img.shape[1]
        num_points = 1000
        for t in np.linspace(0, 2 * math.pi, num_points):
            x_rot = a_rot* math.cos(t) + h_rot
            y_rot = b_rot* math.sin(t) + k_rot
            # 转换回原始坐标系
            # 要转置来逆向旋转
            point = np.dot(rotation_matrix, np.vstack((x_rot, y_rot)))
            # 以预测的R标为原点
            x = int(point[0] + centre[0])
            y = int(- point[1]+ centre[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

        cv2.circle(img, (int(centre[0]), int(centre[1])), 2, (0, 255, 0), -1)
        return img

    def predict_picture(self, img_path, predict_trace=None):
        img=cv2.imread(img_path)
        results=self.model(img)[0]
        print(type(results))
        boxes=results.boxes
        # keypoints=results.keypoints.cpu().numpy()

        for index,box in enumerate(boxes):
            x1,y1,x2,y2=box.xyxy.cpu().numpy()[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            conf=box.conf.cpu().numpy()[0]
            cls=box.cls.cpu().numpy()[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (105, 237, 249), 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 2)
            if cls==0 or cls==2:
                for keypoint in results.keypoints.cpu().numpy()[index].data:
                    if predict_trace:
                        x_list=[]
                        y_list=[]
                    for i, (x, y) in enumerate(keypoint):
                        color_k = [int(x) for x in self.key_point_color[i]]
                        if x != 0 and y != 0:
                            if predict_trace:
                                x_list.append(x)
                                y_list.append(y)
                            cv2.circle(img, (int(x), int(y)), 3, color_k, -1, lineType=cv2.LINE_AA)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img, str(i), (int(x) - 10, int(y) - 10), font, 0.5, (255, 255, 255), 1)
                if predict_trace:
                    size=(img.shape[1],img.shape[0])
                    anchor_points=np.float32([[x_list[0],y_list[0]],[x_list[1],y_list[1]],[x_list[4],y_list[4]],[x_list[3],y_list[3]]])
                    trace=self.find_trace(anchor_points,size)
                    trace_height,trace_width=trace.shape[:2]
                    # 遍历这个二值化图像的所有点，如果是白色那就把同样位置的区域在原图上涂成黑色
                    for y in range(size[1]):
                        for x in range(size[0]):
                            # if y<trace_height and x<trace_width:
                            if trace[y][x] == 255:
                                img[y][x] = [0, 255, 0]
        # for keypoint in keypoints.data:
        #     for i, (x, y) in enumerate(keypoint):
        #         color_k = [int(x) for x in self.key_point_color[i]]
        #         if x != 0 and y != 0:
        #             cv2.circle(img, (int(x), int(y)), 8, color_k, -1, lineType=cv2.LINE_AA)
        #             font = cv2.FONT_HERSHEY_SIMPLEX
        #             cv2.putText(img, str(i), (int(x) - 10, int(y) - 10), font, 0.5, (255, 255, 255), 1)

        (name, suffix) = os.path.splitext(img_path)
        res_path=name+'_pose_predict.jpg'
        print(res_path)
        cv2.imwrite(res_path, img)

    def predict_video(self, video_path, predict_trace=False):
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        (name, suffix) = os.path.splitext(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        res_path=name+self.model_name + '_predict.avi'
        out = cv2.VideoWriter(res_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        try:
            fitting_enable = False
            # 防止在声明前调用
            current_R_point=(0,0)
            counter = 0
            color = None
            if predict_trace:
                draw_enable=False
                fitting_enable=True
                first_radius=None
                # canvas = np.zeros((1000,1000 ), dtype="uint8")
                trace_points=[]
                # cv2.circle(canvas,(500,500),2,255,-1)

            # Process the video frame by frame
            while cap.isOpened():
                canvas = np.full((1000, 1000, 3), 255, dtype=np.uint8)
                cv2.circle(canvas, (500, 500), 2, (0,0,255), -1)
                ret, frame = cap.read()
                if not ret:
                    break
                results=self.model.predict(frame,conf=0.25)[0]
                boxes=results.boxes
                R_points=np.empty((0,2),np.float32)
                # 防止出现只看见非击打目标导致更新异常的问题
                fitting_enable=False
                for index, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf.cpu().numpy()[0]
                    cls = box.cls.cpu().numpy()[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (105, 237, 249), 2)
                    label = f"{cls} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 2)
                    if cls==0 or cls==1:
                        color='red'
                    else:
                        color='blue'
                    # 只标注要击打的目标的的关键点
                    if (cls == 0 or cls == 2) and conf >0.25:
                        # 只有看见目标才预测
                        # 隔点采样有滤波效果
                        if counter%1==0:
                            fitting_enable=True
                        counter += 1
                        # 初始化靶心坐标
                        if predict_trace:
                            centre_x_sum=0
                            centre_y_sum=0
                        for keypoint in results.keypoints.cpu().numpy()[index].data:
                            for i, (x, y) in enumerate(keypoint):
                                color_k = [int(x) for x in self.key_point_color[i]]
                                if x != 0 and y != 0:
                                    cv2.circle(frame, (int(x), int(y)), 2, color_k, -1, lineType=cv2.LINE_AA)
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(frame, str(i), (int(x) - 10, int(y) - 10), font, 0.5, (255, 255, 255), 1)
                                if fitting_enable:
                                    # 前四个点是靶子上的，所有点取平均得到中心点
                                    if i!=4:
                                        centre_x_sum+=x
                                        centre_y_sum+=y
                                    #最后一个点是R的中心
                                    else:
                                        # R_points=np.append(R_points,np.array([[x,y]]),axis=0)
                                        current_R_point=(int(x),int(y))
                    else:
                        x,y=results.keypoints.cpu().numpy()[index].data[0, 0, :]
                        R_points = np.append(R_points, np.array([[x,y]]), axis=0)

                # if fitting_enable and R_points.size != 0:

                if fitting_enable and current_R_point!=None:
                    bullseye = (centre_x_sum / 4, centre_y_sum / 4)

                    bullseye_relative_to_R = (bullseye[0] - current_R_point[0], -bullseye[1] + current_R_point[1])
                    radius = np.linalg.norm(bullseye_relative_to_R)

                    # 将第一次的半径作为椭圆的初始化参数
                    if counter == 1:
                        # 用来初始化椭圆方程
                        first_radius=radius
                        norm_radius = radius / min(frame.shape[0], frame.shape[1])
                        init_A = np.float32([1, 0, 1, 0, 0, -norm_radius ** 2])
                        self.rls = RecursiveLeastSquares(6, initial_A=init_A,delta=1)

                        # 滤波算法
                        self.kalman_filter_2d = KalmanFilter2D()
                        self.kalman_filter_4d= KalmanFilter(4,4,0)

                        #初始化用来获得R点
                        self.get_R_point=ExtractCentre(color)
                        draw_enable=True

                    if draw_enable:

                        current_R_point=self.kalman_filter_2d.correct(current_R_point)
                        current_R_point=(int(current_R_point[0]),int(current_R_point[1]))
                        # 确定R的中心
                        current_R_point=self.get_R_point.get_R_centre(frame,current_R_point,int(radius*0.1))
                        if current_R_point!=None:
                            bullseye_relative_to_R=(bullseye[0] - current_R_point[0], -bullseye[1] + current_R_point[1])
                            # 优化椭圆
                            para, rotation = self.calculate_ellipse(bullseye_relative_to_R, frame.shape[:2])
                            h_rot, k_rot, a_rot, b_rot = para

                            # para=self.kalman_filter_4d.correct(para)
                            self.draw_ellipse(para, rotation, frame, current_R_point)
                            # cv2.circle(frame, (int(current_R_point[0]), int(current_R_point[1])), int(first_radius), (0, 255, 0), 2)

                            # 用来看在这个坐标系到底到底啥样
                            tmp=(int(500 + bullseye[0] - current_R_point[0]), int(500 + bullseye[1] - current_R_point[1]))
                            trace_points.append(tmp)
                            for point in trace_points:
                                 cv2.circle(canvas, point, 2, (0,0,255),-1)
                            self.draw_ellipse(para, rotation, canvas, (500,500))
                            # cv2.imshow('frame',frame)
                            # cv2.waitKey(0)
                cv2.imshow('frame', frame)
                cv2.imshow("canvas", canvas)
                cv2.waitKey(0)
                out.write(frame)
        finally:
            # Release resources
            cap.release()
            out.release()
            print("completed!")

    #由于本赛季可能会在不同位置打符，需要纠正透视变换，所以不能像原来那样直接使用圆形作为轨迹本函数用来获得纠正后的轨迹
    def find_trace(self,points,size):
        trace= np.zeros((334, 334), dtype=np.uint8)
        cv2.circle(trace,(167,167),140,255,3)
        cv2.circle(trace, (167, 167), 3, 255, -1)
        cv2.circle(trace, (167, 0), 3, 255, -1)
        cv2.circle(trace, (194, 27), 3, 255, -1)
        cv2.circle(trace, (140, 27), 3, 255, -1)

        cv2.imshow("origin_mask",trace)

        src= np.float32([[167, 0], [194, 27], [167, 167], [140, 27]])
        dst = points
        pt=cv2.getPerspectiveTransform(src, dst)
        res_trace=cv2.warpPerspective(trace,pt,size)
        cv2.imshow("after_mask",res_trace)

        test_dst = np.float32([[167, 0], [194, 27], [167, 167], [140, 27]])
        test_src =  points
        test_pt=cv2.getPerspectiveTransform(test_src, test_dst)
        test_res_trace=cv2.warpPerspective(res_trace, test_pt, (334, 334))
        cv2.imshow("test",test_res_trace)
        cv2.waitKey(0)
        return res_trace

if __name__ == "__main__":
    # models_path=[r"D:\BuffDetect\best1.pt",r"D:\BuffDetect\best2.pt",r"D:\BuffDetect\best3.pt",r"D:\BuffDetect\best4.pt",r"D:\BuffDetect\best5.pt",r"D:\BuffDetect\best6.pt"]
    # for model in models_path:
    #     predict=Predict_pose_visualize(model_path=model)
    #     # predict.predict_picture(r"D:\Github\RM2025_yolov8\test.png")
    #     predict.predict_video(r"D:\BuffDetect\3se.mp4")
    #
    predict = Predict_pose_visualize(model_path=r"D:\BuffDetect\pose\train8\train8\weights\last.pt")
    predict.predict_video(r"D:\BuffDetect\recordFromNX\8_7_12_video.mp4",True)
    # predict.predict_picture(r"D:\BuffDetect\123123.jpg",True)