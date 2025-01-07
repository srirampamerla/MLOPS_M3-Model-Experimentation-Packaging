# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Run the script to download data
RUN python scripts/download_data.py

# Run the GridSearch and save the model
RUN python scripts/gridsearch.py

# Expose the port for Flask or other services (optional)
EXPOSE 5000

# Default command (can be overridden when running the container)
CMD ["python", "scripts/gridsearch.py"]
