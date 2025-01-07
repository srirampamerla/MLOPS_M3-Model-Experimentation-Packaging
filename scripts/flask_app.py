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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()

        # Ensure that the input data is in the expected format
        required_features = ['feature1', 'feature2', 'feature3', 'feature4']  # Change based on your model's input features
        
        # Check if all required features are present in the input data
        if not all(feature in data for feature in required_features):
            return jsonify({"error": f"Missing required features: {', '.join(required_features)}"}), 400

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
