# Use the official Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Run the script to download data
RUN python scripts/download_data.py

# Run the GridSearch and save the model
RUN python scripts/gridsearch.py

# Expose the port for Flask (or other services) to be accessible
EXPOSE 5000

# Default command to run Flask app after model training is complete
CMD ["python", "scripts/flask_app.py"]  # Assuming app.py starts the Flask app
