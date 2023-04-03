import os
import cv2
import keyboard
import numpy as np
from PIL import Image
import streamlit as st
import mediapipe as mp
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Webcam Interface", page_icon=":camera:", layout="wide")

def draw_landmarks(image, results):
    mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

def image_process(image, model):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def keypoint_extraction(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])

st.header('Webcam Interface')
st.info('In this section, the sign-language recognition models on alphanumeric characters can be tested.')

st.sidebar.header("Webcam Interface")
st.sidebar.caption('For simplicity and easier hosting, the dataset is recorded with only the right hand.')
st.sidebar.info("Webcam Page lets you test two models trained on alphanumeric characters of ASL.")

choice = st.select_slider('Select a model:', options=['None', 'Alpha', 'Num'])
if choice == 'None':
    st.caption("Select 'Alpha' or 'Num' model to test out alphabets or numbers in ASL respectively.")
if choice == 'Alpha':
    PATH = os.path.join('data/alpha')
    actions = np.array(os.listdir(PATH))
    model = load_model('alpha_model')
if choice == 'Num':
    PATH = os.path.join('data/num')
    actions = np.array(os.listdir(PATH))
    model = load_model('num_model')

if choice == 'Num' or choice == 'Alpha':

    col1, col2 = st.columns(2)
    with col1:
        image = Image.open('./icons/ref.jpg')
        st.image(image, caption='@Credit~ Finlay McNevin', width=375)
    with col2:
        st.error("Press 'l' to clear out the video captions when needed.")
        sentence, keypoints = [' '], []
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access camera.")
            exit()

        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            while run:
                _, image = cap.read()
                results = image_process(image, holistic)
                keypoints.append(keypoint_extraction(results))

                if len(keypoints) == 10:
                    keypoints = np.array(keypoints)
                    prediction = model.predict(keypoints[np.newaxis, :, :])
                    keypoints = []
                    
                    if np.amax(prediction) > 0.9:
                        if sentence[-1] != actions[np.argmax(prediction)]:
                            sentence.append(actions[np.argmax(prediction)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                if keyboard.is_pressed('l'):
                    sentence = [' ']

                textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image.shape[1] - textsize[0]) // 2
                    
                cv2.putText(image, ' '.join(sentence), (text_X_coord, 470), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(image)
            else:
                st.write('')