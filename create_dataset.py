import os
import pickle

import mediapipe as mp
import cv2


'''
1. set up hands detection models
2. initialize the data directory and variables for storing data
3. nested for loop for each letter class and each img letter
4. convert image to rgb
5. variable for processed rbg_img with hands model
6. nested for loop for each hand(should only be 1 hand) for each landmark
7. add x and y coordinates to an array
8. add that array and label to main arrays
9. store in pickle dataset
'''

model = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #model for getting landmarks
DATA_DIR = '../sign-language-detector-python/data'
data = [] #The coordinate pairs for each letter
labels = [] #The letter class

for letter_path in os.listdir(DATA_DIR): # Iterate through class folders ('0', '1')
    for img_path in os.listdir(os.path.join(DATA_DIR, letter_path)): # Process each image in class
        img = cv2.imread(os.path.join(DATA_DIR, letter_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MediaPipe requires RGB, OpenCV loads BGR

        result = model.process(img_rgb) # Extract hand landmarks
        if result.multi_hand_landmarks: # Skip images where no hand detected
            hand_landmarks = result.multi_hand_landmarks[0]
            coord_set = []


            for hand_landmarks in result.multi_hand_landmarks: #Goes through each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    coord_set.append(x)
                    coord_set.append(y)

            data.append(coord_set) #Storing the coord pairs into the data and then the letter class
            labels.append(letter_path)

f = open('data.pickle', 'wb') #Dumping the data into pickle
pickle.dump({'data': data, 'labels': labels}, f)
f.close()


