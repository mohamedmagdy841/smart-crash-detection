from ultralytics import YOLO, solutions
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

model = YOLO("car_acc_aug_V12_16_300/weights/best.pt")
video_path = 'revenue/video.avi'
cap = cv2.VideoCapture(video_path)
cap.set(3,800)
cap.set(4,600)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"carla_speed/carla_output_{timestamp}.mp4"

# Define line points
line_points = [(120, 370), (680, 370)] #[(170, 430), (640, 430)]
region_points = [(170, 470), (640, 470), (640, 430), (170, 430)]

video_writer = cv2.VideoWriter(filename,
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (800, 600))

# Init Object Counter
speed_obj = solutions.SpeedEstimatorCustom(
    reg_pts=line_points,
    names=model.names,
    view_img=True,
)

speed_limit = 5
speed_dict = {}
speed_records = []
speed_records_exceeded = []

while cap.isOpened():
    success, im0 = cap.read()

    if success:
        im0 = cv2.resize(im0, dsize=(800, 600))
        tracks = model.track(im0, persist=True, show=False,conf=0.1, verbose=False, classes=[1])
        im0 = speed_obj.estimate_speed(im0, tracks)

        car_id, speed = speed_obj.return_speed()
        if speed != 0:
            speed_records.append({'id': car_id, 'speed': speed})
        if speed >= speed_limit:
            speed_records_exceeded.append({'id': car_id, 'speed': speed})
        video_writer.write(im0)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Remove duplicate entries based on car ID
unique_speed_records = []
seen_ids = set()
for record in speed_records:
    if record['id'] not in seen_ids:
        unique_speed_records.append(record)
        seen_ids.add(record['id'])

unique_speed_records_exceeded = []
seen_ids_exceeded = set()
for record_exceeded in speed_records_exceeded:
    if record_exceeded['id'] not in seen_ids_exceeded:
        unique_speed_records_exceeded.append(record_exceeded)
        seen_ids_exceeded.add(record_exceeded['id'])


# Save to CSV file
df = pd.DataFrame(unique_speed_records_exceeded)
df.to_csv('carla_speed/speeding_cars_report.csv', index=False)

# Prepare data for plotting
car_ids = [record['id'] for record in unique_speed_records]
speeds = [record['speed'] for record in unique_speed_records]

# Categorize speeds into ranges for pie chart
speed_ranges = {'50-60': 0, '60-70': 0, '70-80': 0, '80+': 0}
for speed in speeds:
    if speed <= 60:
        speed_ranges['50-60'] += 1
    elif speed <= 70:
        speed_ranges['60-70'] += 1
    elif speed <= 80:
        speed_ranges['70-80'] += 1
    else:
        speed_ranges['80+'] += 1

# Data for pie chart
labels = speed_ranges.keys()
sizes = speed_ranges.values()

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
axs[0].bar(car_ids, speeds, color='blue')
axs[0].set_title('Bar Chart of Speeds by Car ID')
axs[0].set_xlabel('Car ID')
axs[0].set_ylabel('Speed (km/h)')
axs[0].axhline(y=speed_limit, color='red', linestyle='--', label=f'Speed Limit: {speed_limit} km/h')
axs[0].legend()
axs[0].tick_params(axis='x', rotation=90)

# Pie chart
axs[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
axs[1].set_title('Pie Chart of Speed Ranges')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
