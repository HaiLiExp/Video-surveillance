# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:27:39 2020
This code get image stream from IP camera, do classification with a trained
model and broadcast the resulting image to another url 
@author: Hai Li
"""
#!/usr/bin/env python

#for broadcasting to url
from flask import Flask, render_template, Response
app = Flask(__name__)

#use the html template for broadcasting to url
@app.route('/')
def index():
    return render_template('index.html')

#to read the byte stream from a url
import requests
IPcam_url = "http://192.168.1.3:8080/shot.jpg"

#for image processing
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import time

#for multithreading computing at the cloud. Since this is IO bound task,
#multithreading is more appropriate than multiprocessing
import threading

#to load the machine learning model in the json format
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

#load the model structure saved in the same folder as this code
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

#load the weights of the model
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        #model = load_model('imdb_mlp_model.h5')
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

#read image, preprocessing, and make classification
def MLpred():
    while True:
        frame_resp = requests.get(IPcam_url)

        if frame_resp.status_code == 200:
            frame_arr  = np.array(bytearray(frame_resp.content),dtype = np.uint8)
            frame = cv2.imdecode(frame_arr,-1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[0: 1080, 0: 1080]
            frame = cv2.resize(frame, (108, 108),interpolation = cv2.INTER_AREA)
            
            pic = np.stack([frame, frame, frame],axis=2)
            pic = np.expand_dims(pic, axis=0)
            images = np.vstack([pic])
            images = images/255    
            classes = loaded_model.predict(images, batch_size=1)
            print(classes[0,0])
            
            if classes[0,0] < 0.5:
                cv2.putText(frame, "NOK", (50, 20),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "OK", (50, 20),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

            cv2.putText(frame, time.strftime("%Y%m%d %H:%M:%S"), (0, 100),
			cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.imshow("IP Webcam",frame)

        else:
            print("frame_resp.status_code != 200")
            frame = None

        return frame

def get_frame():    
    while True:
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=MLpred)
            threads.append(t)
            t.start()
        
        frame=MLpred()
        if frame is None:
            print("frame is none")
            continue  
        flag,encodedImage=cv2.imencode('.jpg',frame)
        if not flag:
            print("encodedimage is none")
            continue        
        yield (b'--frame\r\n' b'Content-Type: text/plain\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

#webpage for image output
@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

"""
To run the app, start the command prompt in windows, go to the folder and 
excute: set FLASK_APP=IPCamPredSteaming.py then excute flask run
"""
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True, 
            use_reloader=False)