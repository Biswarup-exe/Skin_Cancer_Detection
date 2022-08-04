from __future__ import division, print_function
# coding=utf-8
import os

import numpy as np

# Keras

from keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'skin_cancer_detection.h5'
MODEL_PATH = pyd.Daisi("skin_cancer_detection.h5")
# Load your trained model
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

#new

# first import the module
import webbrowser
  
# then make a url variable
url = "http://127.0.0.1:5000/"
# then call the default open method described above
#webbrowser.open(url)

#new wnd


def names(number):
    if number==0:
        return 'MALIGNANT'
    else:
        return 'BENEGIN'


def model_predict(img_path):
    #img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    model=tf.keras.models.load_model('skin_cancer_detection.h5', custom_objects=None, compile=True)
    from matplotlib.pyplot import imshow
    #img = Image.open(r'146.jpg')
    img = Image.open(img_path)
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    #imshow(img)
    preds=(str(res[0][classification]*100) + '% Confidence There Is ' + names(classification))
    #preds = model.predict(x)
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    #app.run(debug=True)
    app.debug = True
    app.run(host="0.0.0.0") #host="0.0.0.0" will make the page accessable
                            #by going to http://[ip]:5000/ on any computer in 
                            #the network.
                            #This URL -->"http://127.0.0.1:5000/" to Try on 
                            #host server

