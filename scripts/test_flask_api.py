import requests
import pandas as pd

def test_flask_api():
    # Define test data for prediction
    data = {'feature1': [5.1], 'feature2': [3.5], 'feature3': [1.4], 'feature4': [0.2]}
    input_data = pd.DataFrame(data)

    # Write the data to a CSV file
    input_data.to_csv('test_data.csv', index=False)

    # Read the data from the file
    read_data = pd.read_csv('test_data.csv')

    # Convert to dictionary and make a prediction request to Flask API
    response = requests.post('http://localhost:5000/predict', json=read_data.to_dict(orient='records'))

    # Print status and response
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    test_flask_api()
