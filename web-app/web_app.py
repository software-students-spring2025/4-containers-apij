from flask import Flask, render_template, Response
import requests
import time

app = Flask(__name__)

# URL of the webcam service on the host machine
# 'host.docker.internal' is a special DNS name that resolves to the host machine from inside Docker 
#WEB APP CONTAINER IS REACHING OUT TO THE ml-client
WEBCAM_URL = "http://asl-model:5001/processed_feed"

@app.route('/')
def index():
    """Home page with webcam stream"""
    return render_template('index.html')

@app.route('/stream')
def stream():
    """Stream the webcam feed from the host machine's webcam service"""
    def generate():
        while True:
            try:
                # Stream the response from the webcam service
                response = requests.get(WEBCAM_URL, stream=True, timeout=10)
                
                # Check if we got a successful response
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=1024):
                        yield chunk
                else:
                    print(f"Error from webcam service: {response.status_code}")
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\nError connecting to webcam\r\n"
                    time.sleep(2)
                    
            except requests.RequestException as e:
                print(f"Error connecting to webcam service: {e}")
                error_msg = f"Could not connect to webcam server at {WEBCAM_URL}. Make sure it's running."
                yield f"--frame\r\nContent-Type: text/plain\r\n\r\n{error_msg}\r\n".encode('utf-8')
                time.sleep(2)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    """Health check endpoint"""
    return "Web app is running!"

if __name__ == '__main__':
    print(f"Starting web app on http://0.0.0.0:5003")
    print(f"Expecting webcam feed from {WEBCAM_URL}")
    app.run(host='0.0.0.0', port=5003, debug=True)