from flask import Flask, request, jsonify
import mlflow.pyfunc
import tensorflow as tf
import io
import numpy as np
import zipfile
import os

app = Flask(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def load_production_model(model_name):
    client = mlflow.tracking.MlflowClient()
    model_version_infos = client.search_model_versions(f"name='{model_name}'")
    production_model = None
    for model_version_info in model_version_infos:
        if model_version_info.current_stage == "Production":
            production_model = model_version_info
            break
    if production_model is None:
        raise Exception(f"No model in production stage found for '{model_name}'.")
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model_name = "FinalModel"
model = load_production_model(model_name)

classes = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        img_file = request.files['image']
        img_bytes = io.BytesIO(img_file.read())
        img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array.numpy()
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]
        return jsonify({img_file.filename: predicted_class})

    elif 'zipfile' in request.files:
        zip_file = request.files['zipfile']
        with zipfile.ZipFile(zip_file, 'r') as z:
            predictions = {}
            for file_info in z.infolist():
                with z.open(file_info) as file:
                    img_bytes = io.BytesIO(file.read())
                    img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    img_array = img_array.numpy()
                    predictions[file_info.filename] = classes[np.argmax(model.predict(img_array))]
        return jsonify(predictions)
    else:
        return jsonify({"error": "No appropriate file part in request"}), 400

@app.route('/', methods=['GET'])
def hello():
    return "Server is running!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
