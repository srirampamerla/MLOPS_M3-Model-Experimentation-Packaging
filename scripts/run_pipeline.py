import os
import subprocess

# Step 1: Download data
def download_data():
    print("Downloading dataset...")
    subprocess.run(['python', 'scripts/download_data.py'], check=True)

# Step 2: Run Grid Search for hyperparameter tuning
def grid_search():
    print("Running GridSearch...")
    subprocess.run(['python', 'scripts/grid_search.py'], check=True)

# Step 3: Save the trained model (GridSearch)
def save_model():
    print("Saving the best model...")
    subprocess.run(['python', 'scripts/save_model.py'], check=True)

# Step 4: Run the Flask app to serve the model
def run_flask_app():
    print("Running Flask app...")
    subprocess.run(['python', 'scripts/flask_app.py'], check=True)

def main():
    download_data()
    grid_search()
    save_model()
    run_flask_app()

if __name__ == "__main__":
    main()
