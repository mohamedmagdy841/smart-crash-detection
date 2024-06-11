from ultralytics import YOLO, solutions
import cv2


model = YOLO("car_acc_aug_V14_16_300/weights/last.pt")
video_path = 'object_detection/object_detection_output.avi'
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
line_points = [(0, 300), (640, 300)]
region_points = [(5, 400), (635, 404), (635, 360), (5, 360)]

video_writer = cv2.VideoWriter("object_counting/object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        tracks = model.track(im0, persist=True, show=False,conf=0.5, verbose=False)

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        cap = cv2.VideoCapture(video_path)
        continue

cap.release()
video_writer.release()
cv2.destroyAllWindows()

