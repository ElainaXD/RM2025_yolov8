import os
import cv2
import torch
import numpy as np

from ultralytics import YOLO


class Predict_pose_visualize:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model=YOLO(model_path)
        self.key_point_color=[[255,0,255],[255,0,255],[255,0,255],[255,0,255],[0,215,255]]
        self.device=device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model_name=os.path.splitext(os.path.basename(model_path))[0]


    def predict_picture(self, img_path):
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
                    for i, (x, y) in enumerate(keypoint):
                        color_k = [int(x) for x in self.key_point_color[i]]
                        if x != 0 and y != 0:
                            cv2.circle(img, (int(x), int(y)), 8, color_k, -1, lineType=cv2.LINE_AA)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img, str(i), (int(x) - 10, int(y) - 10), font, 0.5, (255, 255, 255), 1)

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

    def predict_video(self, video_path):
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
            # Process the video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results=self.model.predict(frame,conf=0.25)[0]
                boxes=results.boxes
                for index, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf.cpu().numpy()[0]
                    cls = box.cls.cpu().numpy()[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (105, 237, 249), 2)
                    label = f"{cls} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 2)

                    # 只标注要击打的目标的的关键点
                    if (cls == 0 or cls == 2) and conf >0.5:
                        for keypoint in results.keypoints.cpu().numpy()[index].data:
                            for i, (x, y) in enumerate(keypoint):
                                color_k = [int(x) for x in self.key_point_color[i]]
                                if x != 0 and y != 0:
                                    cv2.circle(frame, (int(x), int(y)), 6, color_k, -1, lineType=cv2.LINE_AA)
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    cv2.putText(frame, str(i), (int(x) - 10, int(y) - 10), font, 0.5, (255, 255, 255), 1)

                out.write(frame)
        finally:
            # Release resources
            cap.release()
            out.release()
            print("completed!")


if __name__ == "__main__":
    # models_path=[r"D:\BuffDetect\best1.pt",r"D:\BuffDetect\best2.pt",r"D:\BuffDetect\best3.pt",r"D:\BuffDetect\best4.pt",r"D:\BuffDetect\best5.pt",r"D:\BuffDetect\best6.pt"]
    # for model in models_path:
    #     predict=Predict_pose_visualize(model_path=model)
    #     # predict.predict_picture(r"D:\Github\RM2025_yolov8\test.png")
    #     predict.predict_video(r"D:\BuffDetect\3se.mp4")
    #
    predict = Predict_pose_visualize(model_path=r"D:\BuffDetect\train7\train7\weights\best.pt")
    predict.predict_video(r"D:\BuffDetect\recordFromNX\8_7_12_video.mp4")
    # predict.predict_picture(r"C:\Users\wyhao\Desktop\d4f762cc-4ff4-4be1-83a3-6f3f5c8e04bd.png")