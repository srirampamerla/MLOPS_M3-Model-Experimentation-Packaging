import os
import subprocess

def main():
    # Step 1: Check if the dataset exists
    data_path = "data/iris.csv"
    if not os.path.exists(data_path):
        print("Dataset not found. Running download_data.py...")
        subprocess.run(["python", "scripts/download_data.py"])
    else:
        print(f"Dataset already exists at {data_path}.")

    # Step 2: Check if the model exists
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found. Running grid_search.py...")
        subprocess.run(["python", "scripts/grid_search.py"])
    else:
        print(f"Model already exists at {model_path}.")

    # Step 3: Start Flask app
    print("Starting Flask application...")
    subprocess.run(["python", "scripts/flask_app.py"])

if __name__ == "__main__":
    main()
