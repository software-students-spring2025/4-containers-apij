import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import requests
from unittest.mock import MagicMock
from flask import Flask, Response, request, jsonify
import base64
import db  # Import our database module

app = Flask(__name__)

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    model = MagicMock()  # fallback for testing


# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Set a single color for all boxes - green
box_color = (0, 255, 0)  # Green in BGR

def process_frame(frame):
    """Process a single frame and return the processed frame with predictions"""
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    prediction_text = None
    
    if results.multi_hand_landmarks:
        # Draw all hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        # Process only the first hand for prediction
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Reset data for each prediction
        data_aux = []
        x_ = []
        y_ = []
        
        # Get coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)
        
        # Normalize and collect features
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))
        
        # Make prediction
        try:
            prediction = model.predict([np.asarray(data_aux)])
            # Get prediction probabilities
            prediction_probs = model.predict_proba([np.asarray(data_aux)])
            # Get confidence score (probability of the predicted class)
            confidence = float(prediction_probs[0][np.argmax(prediction_probs[0])])
            # The prediction is already a string letter
            prediction_text = prediction[0]
            
            # Draw bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
            cv2.putText(frame, f"{prediction_text} ({confidence:.2f})", (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3, cv2.LINE_AA)
            
            # Save prediction to MongoDB
            if prediction_text:
                # Convert frame to base64 for storage
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                db.save_prediction(frame_base64, prediction_text, confidence)
                
        except Exception as e:
            print(f"Prediction error: {e}")
    
    return frame, prediction_text

@app.route('/process_single_frame', methods=['POST'])
def process_single_frame():
    """Process a single frame from an HTTP POST request"""
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    try:
        # Get the frame from the request
        frame_file = request.files['frame']
        frame_bytes = frame_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        # Process the frame
        processed_frame, prediction = process_frame(frame)
        
        # Convert processed frame to base64 for response
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': processed_frame_base64,
            'prediction': prediction if prediction else None
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

def get_frame_from_bytes(frame_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    # Decode image
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def generate_processed_frames():
    """Legacy webcam stream processing (kept for compatibility)"""
    # URL of the webcam service on the host machine
    WEBCAM_URL = "http://host.docker.internal:5002/video_feed"
    
    print(f"Connecting to webcam at {WEBCAM_URL}...")

    try:  
        # Using requests to get the video feed from the host
        response = requests.get(WEBCAM_URL, stream=True) 
        print("CONNECTED TO WEBCAM STREAM")
    
    except Exception as e: 
        print(f"Failed to connect to webcam stream: {e}")  
        return 
    
    # These will be used to extract frames from the multipart response
    boundary = b'frame'
    frame_marker = b'--' + boundary
    
    # Buffer to accumulate bytes
    buffer = b''
    
    for chunk in response.iter_content(chunk_size=1024):  
        
        buffer += chunk
        
        # Look for frame boundaries
        if frame_marker in buffer:
            parts = buffer.split(frame_marker)
            
            # Process all complete frames
            for part in parts[:-1]:
                if b'Content-Type: image/jpeg' in part:
                    # Extract the JPEG data
                    jpeg_parts = part.split(b'\r\n\r\n')
                    if len(jpeg_parts) > 1:
                        jpeg_data = jpeg_parts[1].strip()
                        
                        # Process the frame if we have data
                        if jpeg_data:
                            # Convert bytes to OpenCV frame
                            frame = get_frame_from_bytes(jpeg_data)
                            
                            if frame is not None: 
                                print("FRAME DECODED SUCCESSFULLY: running prediction...")
                                # Apply model to detect sign language
                                processed_frame, _ = process_frame(frame)
                                
                                # Encode processed frame back to JPEG
                                ret, buffer_img = cv2.imencode('.jpg', processed_frame)
                                if ret: 
                                    
                                    print("YIELDING PROCESSED FRAME...")
                                    # Yield the processed frame
                                    frame_bytes = buffer_img.tobytes()
                                    yield (b'--frame\r\n'
                                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Keep the last incomplete part
            buffer = parts[-1]

@app.route('/processed_feed')
def processed_feed():
    """Stream the processed webcam feed with sign language detection"""  
    #debugging tool
    print("Client connected to /processed_feed")
    return Response(generate_processed_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check endpoint"""
    return "ASL model service is running!"

if __name__ == '__main__':
    print("Starting ASL Model Server on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)


def is_valid_landmark(landmark):
    """Utility function to check if a landmark is valid."""
    return (
        hasattr(landmark, 'x') and 
        hasattr(landmark, 'y') and 
        hasattr(landmark, 'visibility')
    )

def get_landmark_coords(landmark):
    """Return (x, y) coordinates from a landmark object."""
    return (landmark.x, landmark.y)

def run_server():
    print("Starting ASL Model Server on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

if __name__ == '__main__':
    run_server()
