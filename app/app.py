from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
import os

import tensorflow as tf 
from PIL import Image
import numpy as np
import json
import requests
import math

app = Flask(__name__)
Bootstrap(app)
"""
Constants
"""
SIZE=28
MODEL_URI = 'http://localhost:8501/v1/models/digits:predict'
# MODEL_URI = 'http://127.0.0.1:8501/v1/models/digits:predict'

"""
Utility Functions
"""
def logit2prob(logit):
    odds = math.exp(logit)
    prob = odds / (1 + odds)
    return prob

def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, color_mode="grayscale", target_size=(SIZE,SIZE)
    )
    # convert image to array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # reshape dimensions [1,28,28,1]
    image = image.reshape(1,SIZE,SIZE,1)
    # image = np.expand_dims(image, axis=0)
    
    # prepare as pixel data
    image = image.astype('float32')
    image = image / 255.0

    # create json object to send to model server
    data = json.dumps({
        'instances': image.tolist()
    })
    # make a POST and get a response back
    response = requests.post(MODEL_URI, data=data.encode()) # default UTF-8 encoding
    result = json.loads(response.text)
    predict_array = result['predictions'][0]
    prediction = np.argmax(result['predictions'][0])
    predict_logit = np.max(result['predictions'])
    lp = logit2prob(predict_logit)
    predict_proba = round(lp, 3)

    return prediction, predict_proba, predict_array

"""
Routes
"""
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # save uploaded image
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)
            prediction, predict_proba, predict_array = get_prediction(image_path) 
            print("\n")
            print('Prediction Logits Array: ',predict_array)
            print("\n")
            print('Predicted Label: {0} | Predicted Probability: {1:.2f}'.format(prediction,predict_proba))
            print("\n")
            result = {
                'prediction': prediction,
                'predict_proba': predict_proba,
                'image_path': image_path
            }
            return render_template('show.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', debug=True, port=8501)