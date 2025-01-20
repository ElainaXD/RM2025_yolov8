import cv2
import numpy as np
import os


def transform_video(input_path, output_path=None, vertical_strength=0, horizontal_strength=0):
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 如果output_path为None，则生成一个新的输出路径
    if output_path is None:
        input_dir, input_filename = os.path.split(input_path)
        filename, file_extension = os.path.splitext(input_filename)
        output_filename = f"{filename}_perspective_trans{file_extension}"
        output_path = os.path.join(input_dir, output_filename)

    # 定义输出视频的编码和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 根据垂直和水平强度调整透视变换的目标点
    vertical_offset = height * vertical_strength
    horizontal_offset = width * horizontal_strength

    # 定义透视变换的源点和目标点
    src_points = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
    dst_points = np.float32([
        [horizontal_offset, height + vertical_offset],
        [width - horizontal_offset, height + vertical_offset],
        [horizontal_offset, -vertical_offset],
        [width - horizontal_offset, -vertical_offset]
    ])

    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 读取视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 应用透视变换
        transformed_frame = cv2.warpPerspective(frame, perspective_matrix, (width, height))

        # 写入输出视频
        out.write(transformed_frame)

        # 显示结果
        cv2.imshow('Transformed Frame', transformed_frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    input_path = r"D:\BuffDetect\recordFromNX\short.mp4"
    vertical_strength = -0.2  # 从下向上看的强度
    horizontal_strength = 0.1  # 从左向右看的强度
    transform_video(input_path, vertical_strength=vertical_strength, horizontal_strength=horizontal_strength)