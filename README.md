![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

![ML Client - CI](https://github.com/software-students-spring2025/4-containers-apij/actions/workflows/ml-client.yml/badge.svg?branch=main)


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

Follow these steps to get your development environment up and running:

---

## 1. install Python 3.8 (using `pyenv` if needed)

## 2. Setup pipenv with Python 3.8  

Ensure that pipenv is using Python 3.8. If it is not, reinitialize it: 

```bash
pipenv --rm
pipenv --python 3.8
```

We can verify that it works by running: 

```bash
pipenv --py` # this should show a Python 3.8 path 
```

## 3. Navigate to the root project directory 

```bash
cd path-to-project-4-containers-apij
```

## 4. Install Dependencies from Pipfile

```bash
pipenv install
```

## 5. Activate the Pipenv Shell 

```bash
pipenv shell 
```

## 6. Run the Project with Docker Compose 

```bash
docker compose up --build
```

## 7. Access the Application: 

Once the Docker build is complete, open your browser and go to: 

http://localhost:5003


# MongoDB Setup (Required) 
Our system uses a MongoDB container. To run it independently: 
```bash
docker run --name mongodb -d -p 27017:27017 mongo
```

Ensure MongoDB is running before launching the full system with Docker. 
