import numpy as np
import mlflow.sklearn
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.datasets import load_iris

# Initialize the Flask application
app = Flask(__name__)
CORS(app) #Allow the app to handle cross-origin requests, making it accessible from different domains

# Load the model
model_uri = "runs:/1fbf46f45ad64559928ec1290c8e975d/model"  
model = mlflow.sklearn.load_model(model_uri)

# Load the iris dataset to get the target names
iris = load_iris()

#Define the Prediction Endpoint
@app.route('/invocations', methods=['POST']) #This decorator specifies that the predict function should handle POST requests sent to the /invocations URL.
def predict():
    try:
        # Get the data from the request
        data = request.json["instances"]
        print(f"Received data: {data}")  
        data = np.array(data) #Convert the data into numpy array
        
        # Make predictions
        predictions = model.predict(data)
        predicted_classes = [iris.target_names[p] for p in predictions]

        # Create a response
        response = {
            "predictions": predicted_classes #Constructs a JSON response containing the predicted class names.
        }
        print(f"Response data: {response}")  
        return jsonify(response)  #Converts the response dictionary to a JSON object and returns it.
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777) #Starts the Flask application on host 0.0.0.0 and port 7777.
