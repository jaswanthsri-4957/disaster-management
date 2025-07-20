from ultralytics import YOLO
import cvzone
import cv2
import math
import requests
import urllib
import cv2
import numpy as np
import winsound
import os
from datetime import datetime
import requests
import urllib
import threading
import time
import urllib.request
r_link='https://api.thingspeak.com/channels/369231/fields/1/last?results=2'

def eqs():
        f=urllib.request.urlopen(r_link)
        pr1 = (f.readline()).decode()
        print(pr1)
        return pr1

# ESP32-CAM IP address
esp32cam_url = 'http://192.168.226.82/cam-mid.jpg'


# Function to fetch images from ESP32-CAM
def get_esp32cam_image():
    try:
        response = requests.get(esp32cam_url, timeout=10)
        if response.status_code == 200:
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            return img
    except Exception as e:
        print(f"Error fetching image from ESP32-CAM: {str(e)}")
    return None
#cap = cv2.VideoCapture(0)
model = YOLO('fire_model.pt')

# Reading the classes
classnames = ['fire']

while True:
    #ret, frame = cap.read()
    frame = get_esp32cam_image()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Getting bbox,confidence and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            es=eqs()
            print('EQ:'+str(es))
            if confidence > 70:


                print('fire detected................................................................................................................')
                winsound.Beep(1000, 1000)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
