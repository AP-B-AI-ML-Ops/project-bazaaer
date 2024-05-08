from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)

# # Load the model from MLflow
# model_name = "FinalModel"
# model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     # Assuming data is in the correct format that your model expects
#     predictions = model.predict(data)
#     return jsonify(predictions)

@app.route('/api', methods=['GET'])
def hello():
    return "Server is running!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=9696)