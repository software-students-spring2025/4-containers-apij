from datetime import datetime
import os
from flask import Flask, render_template, jsonify, request, Response, make_response
from pymongo import MongoClient
from dotenv import load_dotenv
import cv2
import numpy as np
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/'))
db = client.asl_detector
detections = db.detections

# Store the latest frame
latest_frame = None

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent ASL detections from the database."""
    recent_detections = list(detections.find(
        {},
        {'_id': 0}  # Exclude MongoDB _id field
    ).sort('timestamp', -1).limit(50))
    return jsonify(recent_detections)

@app.route('/api/detections', methods=['POST'])
def add_detection():
    """Add a new ASL detection to the database."""
    data = request.get_json()
    if not data or 'sign' not in data:
        return jsonify({'error': 'Missing sign data'}), 400
    
    detection = {
        'sign': data['sign'],
        'confidence': data.get('confidence', 1.0),
        'timestamp': datetime.utcnow()
    }
    
    result = detections.insert_one(detection)
    return jsonify({'message': 'Detection added', 'id': str(result.inserted_id)}), 201

@app.route('/api/frame', methods=['POST'])
def receive_frame():
    """Receive and store the latest frame from the ML client."""
    global latest_frame
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    try:
        file = request.files['frame']
        frame_bytes = file.read()
        nparr = np.frombuffer(frame_bytes, np.uint8)
        latest_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return jsonify({'message': 'Frame received'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/frame', methods=['GET'])
def get_frame():
    """Get the latest frame as a JPEG image."""
    global latest_frame
    if latest_frame is None:
        return jsonify({'error': 'No frame available'}), 404
    
    try:
        # Encode frame with quality 80 for better performance
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', latest_frame, encode_param)
        frame_bytes = buffer.tobytes()
        
        # Create response with proper headers
        response = make_response(frame_bytes)
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get statistics about ASL detections."""
    total_detections = detections.count_documents({})
    signs_count = detections.aggregate([
        {'$group': {'_id': '$sign', 'count': {'$sum': 1}}}
    ])
    signs_count = list(signs_count)
    
    return jsonify({
        'total_detections': total_detections,
        'signs_count': signs_count
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
