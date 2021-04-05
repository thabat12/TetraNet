from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import tensorflow as tf
import joblib
import os
import numpy as np
from numpy import asarray
from keras.models import load_model
from PIL import Image, ImageOps
import json


def init():
    global model
    global message

    # print message to inspect any errors

    try:
        message = 'model loaded'
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'trees-v1.h5')
        message = model_path
        model = load_model(model_path)
    except:
        message = 'model loading failed'

@rawhttp
def run(request):
    
    if request.method == 'GET':
        # GET method to test connection
        return AMLResponse(json.dumps('Connection Confirmed'), 300)
    elif request.method == 'POST':
        # For this demonstration, we will be recieving binary image data
        file_bytes = request.files['image']
        image = Image.open(file_bytes)
        image = ImageOps.grayscale(image)
        image = image.resize((128,128))
        image = asarray(image)
        image = np.expand_dims(image, 0)
        # image will be passed on to the predict function as a numpy array
        mask = predict_frame_from_numpy(image)
        # return a list of serializable data for the json response
        return AMLResponse(json.dumps(mask), 200)

def predict_frame_from_numpy(arr):
    mask = model.predict(arr, batch_size= 1)
    mask = np.squeeze(mask, 0)
    mask = tf.argmax(mask, axis= -1)
    mask = np.array(mask).astype('uint8')
    mask = Image.fromarray(mask)
    mask = np.array(mask).tolist()
    return mask
