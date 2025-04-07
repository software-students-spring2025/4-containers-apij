![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)
![ML Client CI](Link will go here)
![Web App CI](Link will go here)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

# ASL Recognition App

## Overview 
This project is a containerized application that recognizes ASL hand signals using a machine learning model. 

## Team 
- [Aaqila Patel](https://github.com/aaqilap)

- [Isha Gopal](https://github.com/ishy04)

- [Preston Lee](https://github.comprestonglee0805)

- [Jennifer Huang](https://github.com/jennhng)

## Setup Instructions
### 1. Clone the repository 
```bash
git clone https://github.com/software-students-spring2025/4-containers-apij.git
cd 4-containers-apij
```

### 2. Navigate to the machine learning client 
```bash
cd machine-learning-client
```

### 3. Install the dependencies 
```bash
pipenv install 
```

### 4. Activate the virtual environment 
```bash
pipenv shell
```

### 5. Run the following scripts in order. After training, the app will open webcam and predict hand signals. Press 'Q' to quit the inference session. 
```bash
python create_dataset.py
python train_classifier.py
python inference_classifier.py
```

### 6. Running the Web Application
To run the web interface for viewing the ASL detection:

```bash
# Start MongoDB and web app containers
docker-compose up -d

# The web interface will be available at:
# http://localhost:5001
```

### 7. Running the ML Client with Web App
With the web app running:

```bash
# In the machine-learning-client directory
python3 main.py
```

You should now see:
- Live webcam feed in your browser at http://localhost:5001
- Real-time sign predictions
- Detection statistics and history

### 8. Troubleshooting
If the video feed is not working:
- Ensure Docker containers are running: `docker-compose ps`
- Check container logs: `docker-compose logs web-app`
- Make sure your webcam isn't being used by another application
- Try restarting the ML client: `python3 main.py`





