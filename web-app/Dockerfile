FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Make templates folder and port
RUN mkdir -p templates
EXPOSE 5003

# Run the application
CMD ["python", "web_app.py"]