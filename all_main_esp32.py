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

import time
import argparse
import imutils
import cv2

import subprocess
import _thread
import os






ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=8000, help="minimum area size")
args = vars(ap.parse_args())

firstFrame1 = None
firstFrame = None



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
cap = cv2.VideoCapture(0)
model = YOLO('fire_model.pt')

# Reading the classes
classnames = ['fire']
text = "Normal"
text1 = "Landslide"
while True:
    frame = get_esp32cam_image()
   
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)
    es=eqs()
    print('EQ:'+str(es))
    if(int(es)>0):
        print('Earthquake Detected..')
        winsound.Beep(500, 500)
    # Getting bbox,confidence and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])

            if confidence > 10:


                print('fire detected................................................................................................................')
                winsound.Beep(1000, 1000)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)


    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
            firstFrame = gray
            continue
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)



  

    
    for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                    continue
            (x, y, w, h) = cv2.boundingRect(c)
            #print x,y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = text1+str(len(cnts))


    cv2.putText(frame, " Area Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
   




    cv2.imshow("OBJ Feed", frame)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
            break
    if key == ord("r"):
            (grabbed, frame) = cap.read()
            text = "Unoccupied"

            if not grabbed:
                    break

            frame = imutils.resize(frame, width=500)
    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            firstFrame = gray




    cv2.imshow('frame', frame)
    cv2.waitKey(1)
