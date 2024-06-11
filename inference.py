import cv2
from ultralytics import YOLO
from multiLineText import arabic_text
from time import strftime
from datetime import datetime

text1 = "Follow the road instructions"
text2 = "Slow down, accident on the road"
text3 = "التزم بتعليمات الطريق"
text4 = "خفف السرعة حادث على الطريق"

model_custom = YOLO('car_acc_aug_V14_16_300/weights/best.pt', task='detect')
video_path = 'object_detection/object_detection_output.avi'
cap = cv2.VideoCapture(video_path)
cap.set(3, 640) # width
cap.set(4, 480) # height
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"object_detection/object_detection_output_{timestamp}.mp4"
video_writer = cv2.VideoWriter(filename,
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model_custom(frame, conf=0.5, verbose=False)
        arabic_img = arabic_text(text3, (255,255,255), 180, 100, size=70)
        cv2.putText(arabic_img, text1, (180, 300), cv2.FONT_HERSHEY_DUPLEX,
                    1.1,
                    color=(255,255,255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

        for r in results:
            for c in r.boxes.cls:
                if int(c) == 0:
                    arabic_img = arabic_text(text4, (0,0,255), 180, 100, size=45)
                    cv2.putText(arabic_img, text2, (180, 300), cv2.FONT_HERSHEY_DUPLEX,
                                0.9,
                                color=(0, 0, 255),
                                thickness=2,
                                lineType=cv2.LINE_AA)
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, strftime("%H:%M:%S"), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 2, cv2.LINE_AA)


        window_name = "model"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(window_name, 100, 50)
        cv2.resizeWindow(window_name, 1000, 720)
        cv2.resize(annotated_frame, dsize=(1000, 720))
        cv2.imshow(window_name, annotated_frame)

        video_writer.write(annotated_frame)

        # To display text on another screen
        # cv2.namedWindow("Warning", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Warning", 1300, 700)
        # cv2.moveWindow("Warning", -1300, 50)
        # cv2.imshow("Warning", arabic_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        cap = cv2.VideoCapture(video_path)
        continue

cap.release()
cv2.destroyAllWindows()
video_writer.release()
