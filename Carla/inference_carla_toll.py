from ultralytics import YOLO, solutions
import cv2
from datetime import datetime

model = YOLO("yolov8x.pt")
video_path = 'carla/video1.mp4'
cap = cv2.VideoCapture(video_path)
cap.set(3,640)
cap.set(4,480)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"carla/carla_output_{timestamp}.mp4"

# Define line points
line_points = [(0, 400), (800, 400)]
region_points = [(170, 470), (640, 470), (640, 430), (170, 430)]

video_writer = cv2.VideoWriter(filename,
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (800, 600))

# Init Object Counter
counter = solutions.ObjectCounterCustom(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)


while cap.isOpened():
    success, im0 = cap.read()

    if success:
        im0 = cv2.resize(im0, dsize=(800, 600))
        tracks = model.track(im0, persist=True, show=False,conf=0.2, verbose=False, classes=[2,3,5,7,1])
        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        cap = cv2.VideoCapture(video_path)
        continue

cap.release()
video_writer.release()
cv2.destroyAllWindows()
