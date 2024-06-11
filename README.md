# Smart Crash Detection and Emergency Assistance System

This project aims to raise awareness about road safety by implementing creative technologies to detect vehicle crashes using camera, and provide immediate assistance by alerting emergency services.

## Features

- Real-time crash detection using mounted camera
- Automatic alert to emergency contacts with location details
- Detecting cars that exceed the speed limit

## 1-Data Collection
+ For the car/accident detection model to work, we have to first collect data of the classes to be detected.

<p align="center">
  <img width="700" src="https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/55c17c58-c47f-4e5a-b90d-b08908704fae">
</p>

## 2-Data Annotation
+  Data annotation is the process of labeling data available in a video, image or text. 
 The data is manually labeled, so that models can easily comprehend a given data 
 source. Using: [roboflow.com](roboflow.com)

<p float="left">
  <img src="https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/ca7ba8a1-3075-494d-8121-60e6280db2a4" width="485" height="440" >
  <img src="https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/04c8201d-d7d0-4a9b-b1d7-13e6cf495a58" width="485" height="440" >
</p>

<br clear="right"/>

## 3-Training the model
+ We split the data into 90% Training, 5% Validation, 5% Testing.
+ Using Pre-trained model and yolov8n weights. Some relevant metrics are shown in the figure:

<p float="left">
  <img src="https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/88cc2e02-f510-4b63-aa6e-9bea3e1c7872" width="500" height="400" >
  <img src="https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/3027fceb-25f2-4261-857e-4f37620ef858" width="750" height="400" >
</p>

<br clear="right"/>

## Example of accident detection model at work:

https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/a09ead5b-f137-4a56-abc4-98943ba11803

## Example of speed estimation on Carla simulator:

https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/56a3b8d1-7996-4622-b057-5161db848a13

This chart shows the details of the previous video:

<p align="center">
  <img width="700" src="https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/1f32895b-fd7f-4bef-8001-3607b61331d8">
</p>

## Example of speed estimation on our cars:

https://github.com/mohamedmagdy841/smart-crash-detection/assets/64127744/cd7b0016-cefd-4674-ab2e-825ee7bf294c

You will find other videos in the results folder.

## Acknowledgements

- Special thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for providing essential functionalities.
- Thanks to all contributors and testers for their invaluable feedback.






