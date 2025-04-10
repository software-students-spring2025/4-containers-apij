from flask import Flask, render_template, Response, request, jsonify
import requests
import time
import base64
import os
import numpy as np
import cv2
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create the static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

app = Flask(__name__)
app.static_folder = 'static'

# URL of the ASL model service
ASL_MODEL_URL = "http://asl-model:5001/process_single_frame"

# Initialize MongoDB connection with retry mechanism
mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/')
mongodb_connected = False
client = None
db = None

# Retry MongoDB connection
max_retries = 5
retry_delay = 5  # seconds

for attempt in range(max_retries):
    try:
        print(f"Connecting to MongoDB (attempt {attempt+1}/{max_retries})...")
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.server_info()
        db = client.asl_db  # Use the same database name as the ML client
        mongodb_connected = True
        print("Successfully connected to MongoDB")
        break
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"Error connecting to MongoDB: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Falling back to demo data mode")
            mongodb_connected = False
    except Exception as e:
        print(f"Unexpected error connecting to MongoDB: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("Max retries reached. Falling back to demo data mode")
            mongodb_connected = False

# Demo data store for when MongoDB is not available
class DemoDataStore:
    def __init__(self):
        self.signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.demo_data = []
        self.initialize_demo_data()
    
    def initialize_demo_data(self):
        # Generate some random demo data
        now = datetime.now()
        for i in range(30):
            sign = np.random.choice(self.signs)
            confidence = np.random.uniform(0.7, 0.99)
            timestamp = now - timedelta(minutes=i)
            self.demo_data.append({
                'sign': sign,
                'confidence': confidence,
                'timestamp': timestamp
            })
    
    def get_recent_detections(self, minutes=30):
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [d for d in self.demo_data if d['timestamp'] >= cutoff_time]
    
    def add_detection(self, sign, confidence):
        self.demo_data.append({
            'sign': sign,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        # Keep only the last 100 detections
        if len(self.demo_data) > 100:
            self.demo_data = self.demo_data[-100:]

# Initialize demo data store
demo_store = DemoDataStore()

@app.route('/')
def index():
    """Home page with webcam stream and statistics"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get statistics about ASL detections from MongoDB."""
    try:
        if mongodb_connected:
            # Get detections from the last 30 minutes
            cutoff_time = datetime.now() - timedelta(minutes=30)
            detections = list(db.predictions.find({'timestamp': {'$gte': cutoff_time}}))
            
            if not detections:
                # If no data in MongoDB, use demo data
                detections = demo_store.get_recent_detections()
        else:
            # Use demo data if MongoDB is not connected
            detections = demo_store.get_recent_detections()
        
        # Calculate statistics
        total_detections = len(detections)
        unique_signs = len(set(d['prediction'] for d in detections)) if detections else 0
        avg_confidence = 0.95  # Default confidence since ML client doesn't provide it
        last_detection = detections[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if detections else "No detections yet"
        
        # Create frequency chart
        if detections:
            df = pd.DataFrame(detections)
            sign_counts = df['prediction'].value_counts().reset_index()
            sign_counts.columns = ['sign', 'count']
            
            freq_fig = px.bar(sign_counts, x='sign', y='count', 
                             title='ASL Sign Detection Frequency',
                             labels={'sign': 'ASL Sign', 'count': 'Number of Detections'})
            
            # Create timeline chart
            df['hour'] = df['timestamp'].dt.hour
            hourly_counts = df.groupby('hour').size().reset_index(name='count')
            
            timeline_fig = px.line(hourly_counts, x='hour', y='count',
                                  title='ASL Sign Detections by Hour',
                                  labels={'hour': 'Hour of Day', 'count': 'Number of Detections'})
            
            charts = {
                'frequency': json.loads(freq_fig.to_json()),
                'timeline': json.loads(timeline_fig.to_json())
            }
        else:
            charts = {
                'frequency': {'data': [], 'layout': {}},
                'timeline': {'data': [], 'layout': {}}
            }
        
        return jsonify({
            'stats': {
                'total_detections': total_detections,
                'unique_signs': unique_signs,
                'avg_confidence': avg_confidence,
                'last_detection': last_detection
            },
            'charts': charts
        })
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({
            'stats': {
                'total_detections': 0,
                'unique_signs': 0,
                'avg_confidence': 0,
                'last_detection': "Error retrieving data"
            },
            'charts': {
                'frequency': {'data': [], 'layout': {}},
                'timeline': {'data': [], 'layout': {}}
            }
        })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a frame from the browser camera"""
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    # Get the frame from the request
    frame_file = request.files['frame']
    
    try:
        # Read the image data
        frame_bytes = frame_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        # Process the frame with the ASL model service
        processed_frame, prediction = process_with_model(frame)
        
        # Store the detection in MongoDB
        if prediction and prediction != "Error":
            if mongodb_connected:
                # Convert frame to base64 for storage
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                db.predictions.insert_one({
                    "frame_data": frame_base64,
                    "prediction": prediction,
                    "timestamp": datetime.utcnow()
                })
            else:
                demo_store.add_detection(prediction, 0.95)
        
        # Convert processed frame to base64 for sending back to browser
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': processed_frame_base64,
            'prediction': prediction
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

def process_with_model(frame):
    """Process the frame locally with the ASL model service"""
    try:
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Send to ASL model service
        files = {'frame': ('frame.jpg', frame_bytes, 'image/jpeg')}
        response = requests.post(ASL_MODEL_URL, files=files, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            processed_frame_bytes = base64.b64decode(data['processed_image'])
            nparr = np.frombuffer(processed_frame_bytes, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            prediction = data['prediction']
            
            return processed_frame, prediction
        else:
            print(f"Error from ASL model service: {response.status_code}")
            # Return the original frame with an error message
            cv2.putText(frame, "Model Error", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return frame, "Error"
            
    except requests.RequestException as e:
        print(f"Error connecting to ASL model service: {e}")
        # Return the original frame with an error message
        cv2.putText(frame, "Connection Error", (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame, "Error"

def extract_prediction_from_frame(frame):
    """Extract the prediction text from the processed frame"""
    # This is a fallback if the model service doesn't return the prediction separately
    return "ASL"

@app.route('/stream')
def stream():
    """Legacy endpoint for direct streaming (not used with browser camera)"""
    return "Browser camera access now used instead", 200

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print(f"Starting web app on http://0.0.0.0:5003")
    app.run(host='0.0.0.0', port=5003, debug=True)

# Triggering CI another time - 2
