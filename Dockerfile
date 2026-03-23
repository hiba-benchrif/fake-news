# Use smaller python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies if required for sklearn/pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY . .

# Set environment variables for Flask
ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV PYTHONUNBUFFERED=1

# Train the ML model to generate model.pkl and vectorizer.pkl before running
RUN python train_model.py

# Expose port
EXPOSE 5000

# Command to run the application using flask cli
CMD ["flask", "run"]
