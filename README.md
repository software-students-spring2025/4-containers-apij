![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg) ![ML Client - CI](https://github.com/software-students-spring2025/4-containers-apij/actions/workflows/ml-client.yml/badge.svg?branch=main) ![Web App - CI](https://github.com/software-students-spring2025/4-containers-apij/actions/workflows/web-app.yml/badge.svg)


# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

# American Sign Language (ASL) Recognition App

## Overview 
This project is a containerized application that recognizes American Sign Language(ASL) hand signals in real time using a machine learning model. It uses computer vision to interpret hand gestures captured via webcam, analyzes them with a trained model, and displays results through a web app. The objective is to make technology more accessible for those who rely on sign language to communicate.

## Team 
- [Aaqila Patel](https://github.com/aaqilap)

- [Isha Gopal](https://github.com/ishy04)

- [Preston Lee](https://github.com/prestonglee0805)

- [Jennifer Huang](https://github.com/jennhng)

## System Architecture
The system consists of three main components:
1. **Machine Learning Client**: Processes webcam feed to detect ASL signs
2. **Web Application**: Provides a user interface for real-time ASL recognition
3. **MongoDB Database**: Stores detection results and statistics

## Setup Instructions
Follow these steps to get your development environment up and running:


### 1. Install Python 3.8 (using `pyenv` if needed)

### 2. Setup pipenv with Python 3.8  
Ensure that pipenv is using Python 3.8. If it is not, reinitialize it: 

```bash
pipenv --rm
pipenv --python 3.8
pipenv --py  # This will show a Python 3.8 path
```

We can verify that it works by running: 
```bash
pipenv --py # This will show a Python 3.8 path 
```

### 3. Navigate to the root project directory 
```bash
cd path-to-project-4-containers-apij
``` 

### 4. Install Dependencies from Pipfile
```bash
pipenv install
```

### 5. Activate the Pipenv Shell 
```bash
pipenv shell
```

### 6. Run the Project with Docker Compose
```bash
docker compose up --build
```

### 7. Access the Application: 
Once the Docker build is complete, open your browser and go to: 
http://localhost:5003

## MongoDB Setup
MongoDB is automatically configured and run through Docker Compose. The database will be initialized with the following settings:
- Database Name: asl_db<br>
- Collection: predictions<br>
- Port: 27017

## Environment Configuration
Create a '.env' file for your secrets/configurations
Example: 
```bash
MONGO_URI=mongodb://localhost:27017
FLASK_SECRET_KEY=your_secret_key
```

## Development Workflow
1. Create a feature branch for your changes
2. Make your changes and commit them
3. Create a pull request
4. Get code review from at least one team member
5. Merge after approval

## Testing
Both the web app and machine learning client have test suites that must pass before merging:
```bash
# Run web app tests
cd web-app
python -m pytest tests/ --cov=web_app --cov-report=term-missing --cov-fail-under=80

# Run ML client tests
cd ../machine-learning-client
python -m pytest tests/ --cov=ml_client --cov-report=term-missing --cov-fail-under=80
```

