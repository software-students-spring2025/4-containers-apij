import cv2
import numpy as np
from PIL import Image
import requests
import time
import io
import mediapipe as mp
import pickle
from datetime import datetime

def initialize_camera():
    """Initialize and configure the camera."""
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera initialized successfully")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    return cap

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
        response = requests.post('http://localhost:5001/api/frame', files=files)
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
        response = requests.post('http://localhost:5001/api/detections', json=data)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending detection: {e}")
        return False

def main():
    try:
        # Initialize camera and hand detection
        cap = initialize_camera()
        hands, mp_draw, model = initialize_hand_detection()
        
        print("Starting video stream...")
        frame_count = 0
        start_time = time.time()
        last_detection_time = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                time.sleep(0.1)
                continue
            
            # Process hand landmarks
            frame, results = process_hand_landmarks(frame, hands, mp_draw)
            
            # Perform sign detection every 2 seconds
            current_time = time.time()
            if current_time - last_detection_time >= 2.0 and model is not None:
                if results.multi_hand_landmarks:
                    # Extract features from hand landmarks
                    features = extract_features(results.multi_hand_landmarks[0])
                    
                    # Make prediction
                    try:
                        prediction = model.predict([features])[0]
                        confidence = model.predict_proba([features])[0].max()
                        
                        # Send detection if confidence is high enough
                        if confidence > 0.7:
                            send_detection(prediction, confidence)
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
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nStopping stream...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        if 'hands' in locals():
            hands.close()
        print("Camera and hand detection released")

if __name__ == "__main__":
    main()
