![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

[![ML Client - CI](https://github.com/aaqilap/4-containers-apij/actions/workflows/ml-client.yml/badge.svg?branch=main)]


![Web App - CI](https://github.com/software-students-spring2025/4-containers-apij/actions/workflows/web-app.yml/badge.svg)


# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

# ASL Recognition App

## Overview 
This project is a containerized application that recognizes ASL hand signals in real time using a machine learning model. It uses computer vision to interpret hand gestures captured via webcam, analyzes them with a trained model, and displays results through a web app. The objective is to make technology more accessible for those who rely on sign language to communicate.

## Team 
- [Aaqila Patel](https://github.com/aaqilap)

- [Isha Gopal](https://github.com/ishy04)

- [Preston Lee](https://github.com/prestonglee0805)

- [Jennifer Huang](https://github.com/jennhng)

## Setup Instructions
### 1. Clone the repository 
```bash
git clone https://github.com/software-students-spring2025/4-containers-apij.git
cd 4-containers-apij
```

### 2. Navigate to the machine learning client 
```bash
cd ml-client
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
# Navigate to the host directory (4-containers-apij). For example: 
cd /Users/ICPC/4-containers-apij
```

### 7. Running the Video Feed

```bash
python3 mac_webcam_server.py
# The video feed will be available at: http://localhost:5002/video_feed
```

### 8. Running the Web App
```bash
# In a new terminal:
docker compose up --build
# The web app will be available at: http://0.0.0.0:5003
```

You should now see:
- Real-time sign predictions
- Detection statistics and history

### 8. Troubleshooting
If the video feed is not working:
- Ensure Docker containers are running: `docker-compose ps`
- Check container logs: `docker-compose logs web-app`
- Make sure your webcam isn't being used by another application





