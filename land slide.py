

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



camera = cv2.VideoCapture(0)
       
time.sleep(2)



firstFrame1 = None
firstFrame = None

while True:

                
        (grabbed, frame) = camera.read()
       
        text = "Normal"
        text1 = "Landslide"

       
        if not grabbed:
                break

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
                (grabbed, frame) = camera.read()
                text = "Unoccupied"

                if not grabbed:
                        break

                frame = imutils.resize(frame, width=500)
        
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                firstFrame = gray


camera.release()
cv2.destroyAllWindows()
