from ultralytics import YOLO

if __name__ == '__main__':
    # net_yaml_path = r'D:\Github\RM2025_yolov8\BuffTraining\BuffYaml\net_sturcture.yaml'
    # data_yaml_path = r'D:\Github\RM2025_yolov8\BuffTraining\BuffYaml\win_kpt.yaml'
    # cfg_yaml_path = r'D:\Github\RM2025_yolov8\BuffTraining\BuffYaml\hyp.win.yaml'

    net_yaml_path = "./BuffYaml/net_sturcture.yaml"
    data_yaml_path = "./BuffYaml/win_kpt.yaml"
    cfg_yaml_path = "./BuffYaml/hyp.win.yaml"
    # 创建YOLO模型实例
    model = YOLO(net_yaml_path)

    # 确保模型被移动到CUDA设备上
    model.to('cuda')

    # 开始训练模型，确保参数设置正确
    model.train(
        epochs=15,
        data=data_yaml_path,
        device='cuda',  # 指定训练设备为CUDA
        batch=32,
        cfg=cfg_yaml_path
    )