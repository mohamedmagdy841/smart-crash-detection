from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO("yolov8n.pt",task='detect')

model.train(data="datasets/Project3_car_accident_aug_V14/data.yaml",epochs=300,batch=16,workers=0,patience=150,
           seed=42,fraction=0.8,dropout=0.1,plots=True,name='car_acc_aug_V14_16_300')