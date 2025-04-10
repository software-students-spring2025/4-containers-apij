document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const videoElement = document.getElementById('video');
    const processedVideoElement = document.getElementById('processedVideo');
    const startButton = document.getElementById('startBtn');
    const stopButton = document.getElementById('stopBtn');
    const statusElement = document.getElementById('status');
    const recognitionElement = document.getElementById('recognition');
    
    // Variables
    let stream = null;
    let isCapturing = false;
    let captureInterval = null;
    const captureRate = 100; // Capture frames every 100ms (10fps)
    
    // Event listeners
    startButton.addEventListener('click', startCamera);
    stopButton.addEventListener('click', stopCamera);
    
    // Start camera function
    async function startCamera() {
        try {
            // Check if navigator.mediaDevices exists
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera access is not supported in this browser. Please try a modern browser like Chrome, Firefox, or Edge.');
            }
            
            // Request camera access
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480
                }
            });
            
            // Set video source
            videoElement.srcObject = stream;
            
            // Update UI
            startButton.disabled = true;
            stopButton.disabled = false;
            statusElement.textContent = 'Camera running, ASL recognition active';
            
            // Start capturing and sending frames
            startCapturing();
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            statusElement.textContent = `Error: ${error.message}`;
            alert(`Camera error: ${error.message}`);
        }
    }
    
    // Stop camera function
    function stopCamera() {
        if (stream) {
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            
            // Stop capturing
            stopCapturing();
            
            // Update UI
            videoElement.srcObject = null;
            startButton.disabled = false;
            stopButton.disabled = true;
            statusElement.textContent = 'Camera stopped';
            recognitionElement.textContent = '';
            processedVideoElement.style.display = 'none';
            videoElement.style.display = 'block';
        }
    }
    
    // Start capturing frames
    function startCapturing() {
        if (isCapturing) return;
        
        isCapturing = true;
        captureInterval = setInterval(() => {
            captureAndSendFrame();
        }, captureRate);
    }
    
    // Stop capturing frames
    function stopCapturing() {
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }
        isCapturing = false;
    }
    
    // Capture and send frame to server
    function captureAndSendFrame() {
        if (!isCapturing || !stream) return;
        
        // Create canvas to capture frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        // Draw video frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to Blob
        canvas.toBlob((blob) => {
            sendFrameToServer(blob);
        }, 'image/jpeg', 0.8); // JPEG format with 80% quality
    }
    
    // Send frame to server
    function sendFrameToServer(blob) {
        const formData = new FormData();
        formData.append('frame', blob);
        
        fetch('/process_frame', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Network response was not ok');
        })
        .then(data => {
            // Update the UI with processed frame and prediction
            if (data.processed_image) {
                processedVideoElement.src = 'data:image/jpeg;base64,' + data.processed_image;
                processedVideoElement.style.display = 'block';
                videoElement.style.display = 'none';
                
                if (data.prediction) {
                    recognitionElement.textContent = data.prediction;
                }
            }
        })
        .catch(error => {
            console.error('Error sending frame:', error);
            statusElement.textContent = `Error: ${error.message}`;
        });
    }

    // Display initial status
    statusElement.textContent = 'Click "Start Camera" to begin';
});