import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template, send_file
import re
import math
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import requests
from io import BytesIO
import json


app = Flask("__name__")
app.secret_key = b'_5#y2Ln\xec]/'

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/cataractseverity", methods=['POST'])
def cataractseverity():
    class_names = ["Mild", "Normal", "Severe"]
    model = load_model(r'severity_detection_model.h5')
    image_url =  request.form['image']
    print(image_url)
    image_url = json.loads(image_url)
    
 ####################Left EYE##############
    print('left',image_url['leftImage'])
    
    left_response = requests.get(image_url['leftImage'])
    
    left_image = np.array(
        Image.open(BytesIO(left_response.content)).convert("RGB").resize((128, 128)) # image resizing
    )

    left_image = left_image/255 # normalize the image in 0 to 1 range

    left_img_array = tf.expand_dims(left_image, 0)

    
    left_pred = model.predict(left_img_array)
    left_predicted_class = class_names[np.argmax(left_pred[0])]
    left_confidence = round(100 * (np.max(left_pred)), 2)
    
   ####################Right EYE##############
    print('left',image_url['rightImage'])
    
    right_response = requests.get(image_url['rightImage'])
    
    right_image = np.array(
        Image.open(BytesIO(right_response.content)).convert("RGB").resize((128, 128)) # image resizing
    )

    right_image = right_image/255 # normalize the image in 0 to 1 range

    right_img_array = tf.expand_dims(right_image, 0)

    
    right_pred = model.predict(right_img_array)
    right_predicted_class = class_names[np.argmax(right_pred[0])]
    right_confidence = round(100 * (np.max(right_pred)), 2)
    '''print('------------------')
    print(pred)
    pred_lst = pred[0].tolist()
    max_val = pred_lst.index(max(pred_lst))
    
    if max_val==0:
        print('The eye has mild cataract')
    elif max_val==1:
        print('The eye is Normal')
    else:
        print('The eye has Severe cataract')'''

    return {
            "leftEye": {
            "class": left_predicted_class,
            "confidence": left_confidence
            },
            "rightEye": {
            "class": right_predicted_class,
            "confidence": right_confidence
            }
    
   
    }
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug = True, port=8080)
