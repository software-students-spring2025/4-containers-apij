import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Process all alphabet directories
for dir_ in sorted(os.listdir(DATA_DIR)):
    # Skip non-directories and hidden files
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)) or dir_.startswith('.'):
        continue
        
    print(f"Processing directory: {dir_}")
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Use only the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize and collect features
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

print(f"Processed {len(data)} images with valid hand landmarks")
print(f"Found labels: {sorted(set(labels))}")

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Data saved to data.pickle")