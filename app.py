import base64
import cv2
import torch
import numpy as np
from flask import Flask, flash, request, redirect, url_for,jsonify

app = Flask(__name__)

@app.route('/')
def entry_point():
    return " This is Facial Recognition API Status -->> Active "

@app.route('/vectors/', methods=[ 'POST'])
def vectors():
    if request.method == 'POST':
        print(1)
        argggu = (request.json)
        try:
            eencodee = argggu['img'].encode("utf-8")
        except:  
            return { 'ID' :str(argggu['ID']), 'message' :  "Error, Didn't Receive the image from payload in Vectorizing" }         
        print(2)
    
        try:
            ff = base64.decodebytes(eencodee)
        except:
            return { 'ID' :str(argggu['ID']), 'message' : "Error, Base64 decoding error in Vectorizing"}
        print(3)

        try:
            jpg_as_np = np.frombuffer(ff, dtype=np.uint8)
            ttee = cv2.imdecode(jpg_as_np, flags=1) 
            fileName = cv2.resize(ttee,(224,224))
            # cv2.imshow("ANY",fileName)
            # cv2.waitKey(0)
        except:
            return { 'ID' :str(argggu['ID']), 'message' :  "Error, Base64 decoded buffer is not an image for Vectorizing"}
        print(4)

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        results = model(fileName)
        # print(results.print())
        # print (results.pandas())
        jssson = results.pandas().xyxy[0].to_json(orient="records")
        print(results.pandas().xyxy[0].to_json(orient="records")) 
        return { 'ID' :str(argggu['ID']), 'message' :jssson}


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    
