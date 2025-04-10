FROM python:3.8-slim

WORKDIR /app 

# Install system dependencies (needed for OpenCV and mediapipe)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Make templates folder and port
RUN mkdir -p templates
EXPOSE 5001

# Run the application
CMD ["python", "asl_model.py"]