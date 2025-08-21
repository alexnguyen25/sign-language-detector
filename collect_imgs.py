import os

import cv2

#Checks and creates the data folder
DATA_DIR = '../sign-language-detector-python/data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#Initialize variables
cam = cv2.VideoCapture(0)
numLetters = 29
numPics = 200
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space', 27: 'backspace', 28: 'clear'
}
label_count = 0

for i in range(numLetters): #For each letter
    if not os.path.exists(os.path.join(DATA_DIR, str(i))): #Creates the path to each letter class
        os.makedirs(os.path.join(DATA_DIR, str(i)))
    count = 0
    while True: #Ready screen before collecting data
        ret, frame = cam.read()
        if not ret:  # Check if frame was captured successfully and logs and skips if not
            print("frame not read")
            continue
        cv2.putText(frame, "press q to start collecting", (100,100),cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 3)
        cv2.putText(frame, labels_dict[label_count], (150, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(30) == ord('q'):
            break


    while count < numPics: #Takes pictures until parameter
        ret, frame = cam.read()
        if not ret:  # Check if frame was captured successfully and logs and skips if not
            print("frame not read")
            continue
        cv2.putText(frame, str(count), (100,100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        cv2.waitKey(30)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), f'{count}.jpg'), frame)# Saves as: data/0/0.jpg, data/0/1.jpg, etc.
        count += 1
    label_count += 1

cam.release()
cv2.destroyAllWindows()


