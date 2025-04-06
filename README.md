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






