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
    data = request.get_json()
    try:
        input_df = pd.DataFrame(data)
        predictions = model.predict(input_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
