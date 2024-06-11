import cv2
from ultralytics import YOLO, solutions

model = YOLO("car_acc_aug_V14_16_300/weights/best.pt")
names = model.names

video_path = 'object_detection/object_detection_output.avi'
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, 250), (640, 250)]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        cap = cv2.VideoCapture(video_path)
        continue

    tracks = model.track(im0, persist=True, show=False, conf=0.5)
    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)


cap.release()
video_writer.release()
cv2.destroyAllWindows()