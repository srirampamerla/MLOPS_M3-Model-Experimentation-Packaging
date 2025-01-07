from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "Welcome to the Random Forest Classifier API!"

import logging

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    app.logger.info(f"Received data: {data}")  # Log the received data
    try:
        input_df = pd.DataFrame(data)
        predictions = model.predict(input_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")  # Log the error
        return jsonify({"error": str(e)}), 400

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([data])

        # Predict using the model
        predictions = model.predict(input_df)

        # Return the prediction result as a JSON response
        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        # Return error message in case of failure
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Running the Flask app, listening on all interfaces for Docker container usage
    app.run(host="0.0.0.0", port=5000)
