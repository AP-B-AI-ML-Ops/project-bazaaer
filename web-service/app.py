from flask import Flask, request, jsonify
import mlflow.pyfunc
import tensorflow as tf
import io
import numpy as np
import zipfile
import psycopg
import os
import atexit
from psycopg.rows import dict_row
from tensorflow.keras import layers

app = Flask(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

CONNECT_STRING = f'host={os.getenv("POSTGRES_HOST")} port={os.getenv("POSTGRES_PORT")} user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASSWORD")}'


def prep_db():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS requests(
        timestamp timestamp,
        mean_r float,
        mean_g float,
        mean_b float,
        height integer,
        width integer,
        label integer
    );
    """

    with psycopg.connect(CONNECT_STRING, autocommit=True) as conn:
        # zoek naar database genaamd 'test' in de metadata van postgres
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='production'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE production;")
        with psycopg.connect(f'{CONNECT_STRING} dbname=production') as conn:
            conn.execute(create_table_query)

def calculate_image_statistics(img_array):
    mean = np.mean(img_array, axis=(0, 1))
    height, width, _ = img_array.shape
    return {
        'mean_r': float(mean[0]),
        'mean_g': float(mean[1]),
        'mean_b': float(mean[2]),
        'height': int(height),
        'width': int(width)
    }

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
prep_db()

# Initialize the global database connection
global_conn = psycopg.connect(f"{CONNECT_STRING} dbname=production", row_factory=dict_row)

# Close the connection when the application exits
atexit.register(global_conn.close)


classes = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        img_file = request.files['image']
        img_bytes = io.BytesIO(img_file.read())
        img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        normalization_layer = layers.Rescaling(1./255)
        img_array = normalization_layer(img_array)

        img_stats = calculate_image_statistics(img_array)

        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array.numpy()
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]
        img_stats['label'] = int(np.argmax(prediction))

        with global_conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO requests (timestamp, mean_r, mean_g, mean_b, height, width, label)
                VALUES (now(), %(mean_r)s, %(mean_g)s, %(mean_b)s, %(height)s, %(width)s, %(label)s)
                """,
                img_stats
            )
            global_conn.commit()

        return jsonify({img_file.filename: predicted_class})

    elif 'zipfile' in request.files:
        zip_file = request.files['zipfile']
        with zipfile.ZipFile(zip_file, 'r') as z:
            with global_conn.cursor() as cursor:
                predictions = {}
                for file_info in z.infolist():
                    with z.open(file_info) as file:
                        img_bytes = io.BytesIO(file.read())
                        img = tf.keras.preprocessing.image.load_img(img_bytes, target_size=(224, 224))
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        normalization_layer = layers.Rescaling(1./255)
                        img_array = normalization_layer(img_array)

                        img_stats = calculate_image_statistics(img_array)
                        
                        img_array = tf.expand_dims(img_array, 0)
                        img_array = img_array.numpy()
                        prediction = model.predict(img_array)
                        predictions[file_info.filename] = classes[np.argmax(prediction)]

                        img_stats['label'] = int(np.argmax(prediction))

                        cursor.execute(
                            """
                            INSERT INTO requests (timestamp, mean_r, mean_g, mean_b, height, width, label)
                            VALUES (now(), %(mean_r)s, %(mean_g)s, %(mean_b)s, %(height)s, %(width)s, %(label)s)
                            """,
                            img_stats
                        )
                        global_conn.commit()

        return jsonify(predictions)
    else:
        return jsonify({"error": "No appropriate file part in request"}), 400

@app.route('/', methods=['GET'])
def hello():
    return "Server is running!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
