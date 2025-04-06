from datetime import datetime
import os
from flask import Flask, render_template, jsonify, request, Response, make_response
from pymongo import MongoClient
from dotenv import load_dotenv
import cv2
import numpy as np
from io import BytesIO
import base64
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB connection
print("Connecting to MongoDB...")
try:
    client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/'))
    # Test the connection
    client.server_info()
    db = client.asl_detector
    detections = db.detections
    print("MongoDB connection established successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    print(traceback.format_exc())
    raise

# Store the latest frame
latest_frame = None
latest_frame_time = None

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent ASL detections from the database."""
    try:
        recent_detections = list(detections.find(
            {},
            {'_id': 0}  # Exclude MongoDB _id field
        ).sort('timestamp', -1).limit(50))
        print(f"Retrieved {len(recent_detections)} detections")
        return jsonify(recent_detections)
    except Exception as e:
        print(f"Error getting detections: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections', methods=['POST'])
def add_detection():
    """Add a new ASL detection to the database."""
    try:
        data = request.get_json()
        print(f"Received detection data: {data}")
        
        if not data or 'sign' not in data:
            print("Missing sign data in request")
            return jsonify({'error': 'Missing sign data'}), 400
        
        detection = {
            'sign': data['sign'],
            'confidence': data.get('confidence', 1.0),
            'timestamp': datetime.utcnow()
        }
        
        result = detections.insert_one(detection)
        print(f"Added detection with ID: {result.inserted_id}")
        return jsonify({'message': 'Detection added', 'id': str(result.inserted_id)}), 201
    except Exception as e:
        print(f"Error adding detection: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/frame', methods=['POST'])
def receive_frame():
    """Receive and store the latest frame from the ML client."""
    global latest_frame, latest_frame_time
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    try:
        file = request.files['frame']
        frame_bytes = file.read()
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode frame")
        
        latest_frame = frame
        latest_frame_time = datetime.utcnow()
        print(f"Received new frame, shape: {frame.shape}")
        return jsonify({'message': 'Frame received'}), 200
    except Exception as e:
        print(f"Error receiving frame: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/frame', methods=['GET'])
def get_frame():
    """Get the latest frame as a JPEG image."""
    global latest_frame, latest_frame_time
    if latest_frame is None:
        return jsonify({'error': 'No frame available'}), 404
    
    try:
        # Check if frame is too old (more than 5 seconds)
        if latest_frame_time and (datetime.utcnow() - latest_frame_time).total_seconds() > 5:
            print("Frame is too old")
            return jsonify({'error': 'Frame is too old'}), 404
        
        # Encode frame with quality 80 for better performance
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', latest_frame, encode_param)
        if buffer is None:
            raise ValueError("Failed to encode frame")
            
        frame_bytes = buffer.tobytes()
        
        # Create response with proper headers
        response = make_response(frame_bytes)
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response
    except Exception as e:
        print(f"Error sending frame: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get statistics about ASL detections."""
    try:
        total_detections = detections.count_documents({})
        signs_count = detections.aggregate([
            {'$group': {'_id': '$sign', 'count': {'$sum': 1}}}
        ])
        signs_count = list(signs_count)
        
        print(f"Stats - Total detections: {total_detections}, Signs count: {signs_count}")
        return jsonify({
            'total_detections': total_detections,
            'signs_count': signs_count
        })
    except Exception as e:
        print(f"Error getting stats: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
