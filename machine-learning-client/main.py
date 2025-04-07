import cv2
import numpy as np
from PIL import Image
import requests
import time
import io
import mediapipe as mp
import pickle
from datetime import datetime
import os
from flask import Flask
import threading

# Get web app URL from environment variable
WEB_APP_URL = os.getenv('WEB_APP_URL', 'http://localhost:5001')

# Initialize Flask app for health check
app = Flask(__name__)

@app.route('/health')
def health_check():
    return {'status': 'healthy'}, 200

def initialize_camera():
    """Initialize and configure the camera."""
    print("Initializing camera...")
    try:
        # Try different camera indices
        camera_indices = [0, 1, -1]  # Try default, secondary, and last camera
        cap = None
        
        for idx in camera_indices:
            print(f"Attempting to open camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                # Test if we can actually read frames
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully opened camera index {idx}")
                    break
                else:
                    print(f"Camera {idx} opened but cannot read frames")
                    cap.release()
                    cap = None
            else:
                print(f"Could not open camera index {idx}")
        
        if cap is None:
            raise RuntimeError("Could not open any camera")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera settings
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print("Camera initialized successfully")
        print(f"Resolution: {actual_width}x{actual_height}")
        print(f"FPS: {actual_fps}")
        
        return cap
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
        raise

def initialize_hand_detection():
    """Initialize MediaPipe hand detection."""
    print("Initializing hand detection...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Load the trained model
    try:
        model_dict = pickle.load(open('model.p', 'rb'))
        model = model_dict['model']
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    return hands, mp_draw, model

def process_hand_landmarks(frame, hands, mp_draw):
    """Process hand landmarks in the frame."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )
    
    return frame, results

def extract_features(hand_landmarks):
    """Extract features from hand landmarks."""
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
    
    return np.asarray(data_aux)

def send_frame(frame):
    """Send frame to web app."""
    try:
        # Resize frame to reduce bandwidth
        frame = cv2.resize(frame, (640, 480))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to JPEG
        img = Image.fromarray(frame_rgb)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=80)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Send to web app
        files = {'frame': ('frame.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f'{WEB_APP_URL}/api/frame', files=files)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending frame: {e}")
        return False

def send_detection(sign, confidence=1.0):
    """Send detection to web app."""
    try:
        data = {
            'sign': sign,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        }
        response = requests.post(f'{WEB_APP_URL}/api/detections', json=data)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending detection: {e}")
        return False

def run_flask():
    """Run the Flask server"""
    app.run(host='0.0.0.0', port=5002)

def main():
    try:
        # Start Flask server in a separate thread
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True  # This ensures the thread will be terminated when the main program exits
        flask_thread.start()
        
        # Initialize camera and hand detection
        cap = initialize_camera()
        hands, mp_draw, model = initialize_hand_detection()
        
        print("Starting video stream...")
        frame_count = 0
        start_time = time.time()
        last_detection_time = 0
        last_frame_time = time.time()
        
        while True:
            current_time = time.time()
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Error: Could not read frame. ret={ret}, frame={'None' if frame is None else 'valid'}")
                if current_time - last_frame_time > 5:  # If no frames for 5 seconds
                    print("No frames received for 5 seconds, attempting to reinitialize camera...")
                    cap.release()
                    cap = initialize_camera()
                    last_frame_time = current_time
                time.sleep(0.1)
                continue
            
            last_frame_time = current_time
            
            # Process hand landmarks
            frame, results = process_hand_landmarks(frame, hands, mp_draw)
            
            # Perform sign detection every 2 seconds
            if current_time - last_detection_time >= 2.0 and model is not None:
                if results.multi_hand_landmarks:
                    print("Hand detected, processing landmarks...")
                    # Extract features from hand landmarks
                    features = extract_features(results.multi_hand_landmarks[0])
                    
                    # Make prediction
                    try:
                        prediction = model.predict([features])[0]
                        confidence = model.predict_proba([features])[0].max()
                        print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
                        
                        # Draw prediction on frame
                        cv2.putText(frame, f"{prediction} ({confidence:.2f})", 
                                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                  (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # Send detection if confidence is high enough
                        if confidence > 0.2:  # Lowered from 0.7 to 0.2
                            if send_detection(prediction, confidence):
                                print(f"Successfully sent detection: {prediction}")
                            last_detection_time = current_time
                    except Exception as e:
                        print(f"Prediction error: {e}")
            
            # Send frame to web app
            if send_frame(frame):
                frame_count += 1
                
                # Print FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"Streaming at {fps:.2f} FPS")
            else:
                print("Failed to send frame to web app")
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
