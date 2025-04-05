#This script is used to collect 

import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Update to include all alphabet letters
number_of_classes = 26
dataset_size = 100

# Create folders for all letters A-Z
for i in range(number_of_classes):
    letter = chr(65 + i)  # ASCII: A=65, B=66, etc.
    class_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print(f"Created directory for letter {letter}")

# Camera setup code
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Collect data for each letter
for i in range(number_of_classes):
    letter = chr(65 + i)
    print(f'Collecting data for letter {letter}')
    
    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue
            
        cv2.putText(frame, f'Ready to collect data for letter {letter}. Press "Q" when ready!', 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Small delay after pressing Q
    time.sleep(1)
    
    # Collect images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
            
        cv2.putText(frame, f'Collecting {counter+1}/{dataset_size} for {letter}', 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Save the image
        cv2.imwrite(os.path.join(DATA_DIR, letter, f'{counter}.jpg'), frame)
        counter += 1
        
        # Small delay between captures
        cv2.waitKey(100)
    
    print(f"Finished collecting data for letter {letter}")
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()