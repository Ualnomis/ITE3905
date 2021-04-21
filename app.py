from flask import Flask, render_template, request
import numpy as np
import tensorflow.keras as keras
import re
import io
from io import BytesIO
import base64
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from io import StringIO

app = Flask(__name__)

model_file = './model_190382372.h5'
global model
model = keras.models.load_model(model_file)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['Get', 'POST'])
def predict():
    # save Image 
    saveImage(request.get_data())
    # using keras to convert image to array and change it to size 28 x 28 same as mnist dataset size
    img = img_to_array(load_img('inputedDigit.png', target_size=(28, 28), color_mode="grayscale")) / 255
    # Expand the shape of an array
    img = np.expand_dims(img, axis=0)
    print(img)
    predictedResult = np.argmax(model.predict(img), axis=-1)[0]
    # try to print result
    print(predictedResult)
    return str(predictedResult)  
    
def saveImage(imgBase64):
    # decode base64
    imageStr = re.search(b'base64,(.*)', imgBase64).group(1)
    # save as inputedDigit.png
    with open('./inputedDigit.png', 'wb') as output:
        output.write(base64.decodebytes(imageStr))

if __name__ == '__main__':
    app.run(host="192.168.50.2", port=5000)