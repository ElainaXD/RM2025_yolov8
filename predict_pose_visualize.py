import os
import cv2
import torch
import numpy as np

from ultralytics import YOLO


class Predict_pose_visualize:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model=YOLO(model_path)
        self.key_point_color=[[255,0,255],[255,0,255],[0,215,255],[255,0,255],[255,0,255]]
        self.device=device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    def predict_picture(self, img_path):
        img=cv2.imread(img_path)
        results=self.model(img)[0]
        print(type(results))
        boxes=results.boxes
        keypoints=results.keypoints.cpu().numpy()

        for box in boxes:
            x1,y1,x2,y2=box.xyxy.cpu().numpy()[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            conf=box.conf.cpu().numpy()[0]
            cls=box.cls.cpu().numpy()[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (105, 237, 249), 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 2)

        for keypoint in keypoints.data:
            for i, (x, y) in enumerate(keypoint):
                color_k = [int(x) for x in self.key_point_color[i]]
                if x != 0 and y != 0:
                    cv2.circle(img, (int(x), int(y)), 8, color_k, -1, lineType=cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(i), (int(x) - 10, int(y) - 10), font, 0.5, (255, 255, 255), 1)

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
        res_path=name+'_pose_predict.avi'
        out = cv2.VideoWriter(res_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        try:
            # Process the video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results=self.model(frame)[0]
                boxes=results.boxes
                keypoints=results.keypoints.cpu().numpy()[0]
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy.cpu().numpy()[0]
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                    conf=box.conf.cpu().numpy()[0]
                    cls=box.cls.cpu().numpy()[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 2)


                for keypoint in keypoints.data:
                    for i, (x, y) in enumerate(keypoint):
                        color_k = [int(x) for x in self.key_point_color[i]]
                        if x != 0 and y != 0:
                            cv2.circle(frame, (int(x), int(y)), 8, color_k, -1, lineType=cv2.LINE_AA)
                            cv2.putText(frame, str(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                out.write(frame)
        finally:
            # Release resources
            cap.release()
            out.release()
            print("completed!")


if __name__ == "__main__":
    predict=Predict_pose_visualize(model_path=r"D:\Github\RM2025_yolov8\BuffTraining\runs\pose\best2.pt")
    predict.predict_picture(r"D:\Github\RM2025_yolov8\test.png")
    # predict.predict_video(r"E:\BaiduNetdiskDownload\12mm_red_dark.mp4")