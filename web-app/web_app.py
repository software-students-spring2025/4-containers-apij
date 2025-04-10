from flask import Flask, render_template, Response, request, jsonify
import requests
import time
import base64
import os
import numpy as np
import cv2

# Create the static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

app = Flask(__name__)
app.static_folder = 'static'

# URL of the ASL model service
ASL_MODEL_URL = "http://asl-model:5001/process_single_frame"

@app.route('/')
def index():
    """Home page with webcam stream"""
    return render_template('index.html')

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
    # We have two options here:

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
    return "Web app is running!"

if __name__ == '__main__':
    print(f"Starting web app on http://0.0.0.0:5003")
    app.run(host='0.0.0.0', port=5003, debug=True)