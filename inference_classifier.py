import pickle

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import draw_landmarks

"""
Sign Language Detector
Detects American Sign Language letters A-Z in real-time using MediaPipe and Random Forest
"""

#Initialize Camera and hand detection
cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

#Load trained model
model_dict = pickle.load(open("model.p", "rb"))
model = model_dict['model']

#Sign mapping
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space', 27: 'backspace', 28: 'clear'
}

#Detection Variables
current_character = ""
text_output = ""
quality_check = []
detection_threshold = 20
prediction_proba = None
confidence = 0

while True:
    ret, frame = cam.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_result = hands.process(img_rgb)
    H, W, _ = frame.shape
    if hands_result.multi_hand_landmarks:
        coord_set = []
        x_ = []
        y_ = []

        #Processing land handmarks
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
              )

            #Extract coordinates
            for i in range(len(hand_landmarks.landmark)):
                x_coord = hand_landmarks.landmark[i].x
                y_coord = hand_landmarks.landmark[i].y
                coord_set.append(x_coord)
                coord_set.append(y_coord)
                x_.append(x_coord)
                y_.append(y_coord)

        #Predicting Sign
        prediction = model.predict([np.asarray(coord_set)])
        predicted_character = labels_dict[int(prediction[0])]
        prediction_proba = model.predict_proba([np.asarray(coord_set)])
        confidence = max(prediction_proba[0]) * 100

        #Add to quality check buffer
        current_character = predicted_character
        quality_check.append(current_character)

        #Coordinates for bounding box UI
        x1 = int(min(x_) * W - 10)
        y1 = int(min(y_) * H - 10)
        x2 = int(max(x_) * W + 10)
        y2 = int(max(y_) * H + 10)

        #Bounding Box UI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #Process Detection Buffer
    if len(quality_check) > detection_threshold:
        if len(set(quality_check)) == 1: #All frames show same sign

            #Check for command
            if not(quality_check[0] == 'backspace' or quality_check[0] == 'clear' or quality_check[0] == 'space'):
                text_output += quality_check[0] #Adds character
            else:
                if quality_check[0] == 'backspace':
                    text_output = text_output[:-1]
                if quality_check[0] == 'clear':
                    text_output = ""
                if quality_check[0] == 'space':
                    text_output = text_output + " "
        quality_check = [] #Reset Detection Buffer

    #Display UI
    cv2.putText(frame, text_output + '|', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Press Q to quit | Hold sign steady for detection", (10, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Chars: {len(text_output)}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow('Sign Language Detection', frame)
    print(confidence)

    #Quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()