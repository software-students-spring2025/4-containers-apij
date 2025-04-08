## Project Structure

- `collect_imgs.py`: Collects gesture images using your webcam and organizes them into folders.
- `create_dataset.py`: Processes and prepares the collected images into training data.
- `train_classifier.py`: Trains a classifier model to recognize different signs.
- `inference_classifier.py`: Runs real-time or batch inference using the trained model. 

---

## Getting Started:  

- navigate into the machine learning file from the file path: 

`cd machine-learning-client`  

- Install all dependencies within the machine-learning-client file: 

`pipenv install` 

- Activate the pipenv shell. Ensure that you are using the proper interpretor and insid ethe proper virtual environment. It should be something like (machine-learning-client-...) 

`pipenv shell`

### Image Collection  

NOTE: I have already done the image collection for training inside the data file (see images). You can ignore this step but here is more information: 

You will be prompted to enter a label of A, B, C, ... Z and the script will activate your webcam to capture images for that label, storing in a separate data file. You will press 'q' to take images for each sign language symbol with your hand for each letter. In your terminal, run: 

`python collect_imgs.py` 

## What to run: 

### Dataset Creation: 
Inside the pipenv shell for the machine-learning client, convert and structure the image data running:  

`python create_dataset.py` 

### Train the Classifier: 
Train the model by running in your terminal: 

`python train_classifier.py` 

### Make Predictions Using the Trained Model:  
The next command will trigger the webcam and show live predictions of the hand. Press 'Q' to quit 

`python inference_classifier.py` 

### Demo Image of Working Script
![Demo Image](demoimage.png)
