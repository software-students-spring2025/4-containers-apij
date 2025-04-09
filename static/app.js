//waits for HTML to be fully downloaded
document.addEventListener('DOMContentLoaded', () => { 

  //get reference to video element that displays feed
  const videoElement = document.getElementById('video'); 

  //refernce to image elemnet displaying processed video with ASL recognition
  const processedVideoElement = document.getElementById('processedVideo'); 

  //starts camera 
  const startButton = document.getElementById('startBtn'); 

  //stops camera
  const stopButton = document.getElementById('stopBtn'); 

  //display status
  const statusElement = document.getElementById('status'); 

  //display ASL sign prediction
  const recognitionElement = document.getElementById('recognition');
  
  // Variables
  let stream = null;
  let isCapturing = false;
  let captureInterval = null;
  const captureRate = 100; // Capture frames every 100ms (10fps)
  
  // Event listeners 

  //calls startCamera
  startButton.addEventListener('click', startCamera); 

  //calls stopCamera
  stopButton.addEventListener('click', stopCamera);
  
  // Start camera function
  async function startCamera() {
      try {
          // Request camera access, if it works, output webcam 640x480
          stream = await navigator.mediaDevices.getUserMedia({  
              //video dimensions
              video: { 
                  width: 640, 
                  height: 480
              }
          });
          
          // Set video source to camera stream
          videoElement.srcObject = stream;
          
          //disable start button as now the camera is on
          startButton.disabled = true; 

          //shows stop button now 
          stopButton.disabled = false; 

          //updates status feed to show camera is running
          statusElement.textContent = 'Camera running, ASL recognition active';
          
          // Start capturing and sending frames
          startCapturing();
          
      }   
      //THROW IF ERRORS WITH CAMERA STREAM
      catch (error) {
          console.error('Error accessing camera:', error);
          statusElement.textContent = `Error: ${error.message}`;
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
});