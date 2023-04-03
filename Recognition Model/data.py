import os
import cv2
import string
import keyboard
import numpy as np
from utils import *
import mediapipe as mp
from itertools import product

actions = np.array(list(string.ascii_lowercase) + '0,1,2,3,4,5,6,7,8,9'.split(',') + 'hello,nice,meet,you,thanks,for,watching,my,name'.split(','))
sequences = 30
frames = 10

PATH = os.path.join('data')
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        if frame == 0: 
            while True:
                if keyboard.is_pressed(' '):
                    break
                _, image = cap.read()

                results = image_process(image, holistic)
                draw_landmarks(image, results)

                cv2.putText(image, 'Recording data for the "{}"; Sequence number {}'.format(action, sequence),
                            (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Press "Space" when you are ready', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                cv2.waitKey(1)

                if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                    break
        else:
            _, image = cap.read()
            results = image_process(image, holistic)
            draw_landmarks(image, results)

            cv2.putText(image, 'Recording data for the "{}"; Sequence number {}'.format(action, sequence),
                        (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('Camera', image)
            cv2.waitKey(1)
        
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
             break

        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

    cap.release()
    cv2.destroyAllWindows()
