import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, control_dim=0):
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

    def predict(self):
        # 预测下一个状态
        return self.kf.predict()

    def correct(self, measurement):
        # 使用新的测量值更新卡尔曼滤波器
        return self.kf.correct(measurement)

    def get_state(self):
        # 获取当前状态估计
        return self.kf.statePost

# 使用卡尔曼滤波器类
state_dim = 6
measurement_dim = 6
kf = KalmanFilter(state_dim, measurement_dim)

# 假设我们有一个包含六个测量值的数组
measurement = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

# 预测下一个状态
predicted_state = kf.predict()
print("预测状态:", predicted_state)

# 使用新的测量值更新卡尔曼滤波器
kf.correct(measurement)

# 打印更新后的状态
print("更新后的状态:", kf.get_state())
