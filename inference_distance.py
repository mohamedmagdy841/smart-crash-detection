import cv2
from ultralytics import YOLO, solutions
from datetime import datetime

model = YOLO("car_acc_aug_V12_16_300/weights/best.pt")
cap = cv2.VideoCapture("video_test2.avi")
cap.set(3,800)
cap.set(4,600)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"distance/{timestamp}.mp4"
video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
dist_obj = solutions.DistanceCalculation(names=model.names,
                                         pixels_per_meter=20,
                                            view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False, verbose=False, conf=0.1)#, classes=[2,3,5,7,1])
    im0 = dist_obj.start_process(im0, tracks)
    video_writer.write(im0)
    if cv2.waitKey(15) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()