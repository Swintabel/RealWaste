from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cv2
import numpy as np
import os

labels = {0: 'Cardboard',
 1: 'Food Organics',
 2: 'Glass',
 3: 'Metal',
 4: 'Miscellaneous Trash',
 5: 'Paper',
 6: 'Plastic',
 7: 'Textile Trash',
 8: 'Vegetation'}



app = Flask(__name__)

# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the full paths to the model files
model_path1 = os.path.join(script_dir, "cnnModel.keras")
model_path2 = os.path.join(script_dir, "HYBModel.keras")

cnn = load_model(model_path1)
hyb = load_model(model_path1)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():

    img = request.files['img']

    img.save("img.jpg")

    image= cv2.imread("img.jpg")

    image = cv2.resize(image,(224,224))

    image_array = np.array(image)

    image_array = preprocess_input(image_array)

    img_array = np.expand_dims(image_array, axis=0)

    pred = cnn.predict(img_array)

    pred2 = hyb.predict(img_array)

    predicted_class = np.argmax(pred, axis =1)

    predicted_class2 = np.argmax(pred2, axis =1)

    pred_label = labels[predicted_class[0]]

    pred_label2 = labels[predicted_class2[0]]

    results = {"CNN+efficientnet_v2 Model :      ": pred_label, "Hybrid Model:    ": pred_label2}

    return render_template("prediction.html", data = results)

if __name__=="__main__":
    app.run(debug=True)
