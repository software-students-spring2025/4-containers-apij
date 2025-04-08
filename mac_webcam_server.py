import cv2
from flask import Flask, Response
import time

app = Flask(__name__)

def get_camera():
    """Initialize and return camera object"""
    camera = cv2.VideoCapture(0)  # Use first webcam
    
    # Give camera time to initialize
    time.sleep(2)
    
    # Set resolution (optional) HELP WITH OPTIMIZATION 
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) 
    camera.set(cv2.CAP_PROP_FPS, 30)

    
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")
        
    return camera

def generate_frames():
    """Generate frames from webcam"""
    camera = get_camera()
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read from camera")
                time.sleep(1)
                continue
                
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Convert to bytes and yield
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        # Release camera when done
        camera.release()

@app.route('/video_feed')
def video_feed():
    """Route to serve webcam video feed"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Simple info page"""
    return """
    <h1>MacOS Webcam Server</h1>
    <p>This server is streaming your webcam at the <a href="/video_feed">/video_feed</a> endpoint.</p>
    <p>The Docker container should be connecting to this stream.</p>
    """

if __name__ == '__main__':
    print("Starting webcam server on http://localhost:5002")
    print("Check http://localhost:5002/video_feed to confirm it's working")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)
    except Exception as e:
        print(f"Error: {e}")