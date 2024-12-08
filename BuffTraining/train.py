from ultralytics import YOLO
# import winsound

if __name__ == '__main__':
    # net_yaml_path = r'D:\Github\RM2025_yolov8\BuffTraining\BuffYaml\net_sturcture.yaml'
    # data_yaml_path = r'D:\Github\RM2025_yolov8\BuffTraining\BuffYaml\win_kpt.yaml'
    # cfg_yaml_path = r'D:\Github\RM2025_yolov8\BuffTraining\BuffYaml\hyp.win3.yaml'

    net_yaml_path = "./BuffYaml/net_sturcture.yaml"
    data_yaml_path = "./BuffYaml/win_kpt.yaml"

    # 创建YOLO模型实例
    model = YOLO(net_yaml_path)

    # 确保模型被移动到CUDA设备上
    model.to('cuda')

    # 开始训练模型，确保参数设置正确
    # cfg_yaml_list=["BuffYaml/hyp.win1.yaml","BuffYaml/hyp.win2.yaml","BuffYaml/hyp.win3.yaml","BuffYaml/hyp.win1.yaml","BuffYaml/hyp.win5.yaml","BuffYaml/hyp.win6.yaml","BuffYaml/hyp.win7.yaml","BuffYaml/hyp.win8.yaml"]
    cfg_yaml_list=["BuffYaml/hyp.win2.yaml"]
    for path in cfg_yaml_list:
        cfg_yaml_path = path
        model.train(
            epochs=30,
            data=data_yaml_path,
            device='cuda',  # 指定训练设备为CUDA
            batch=32,
            cfg=cfg_yaml_path
        )
