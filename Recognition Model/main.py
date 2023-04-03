import os
import cv2
# import torch
import keyboard
import numpy as np
from utils import *
import mediapipe as mp
from tensorflow.keras.models import load_model
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# attmodel = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
# tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
# attmodel = attmodel.to(device)

print('Choose one of the following options:/n')
print('1: Main')
print('2: Num')
print('3: Alpha')
print('4: Gesture')
print('0: Exit')
choice = 1

while choice != 0:
    choice = int(input('Enter:\t'))
    if choice == 1:
        PATH = os.path.join('data')
        actions = np.array(os.listdir(PATH))
        model = load_model('model')

    elif choice == 2:
        PATH = os.path.join('specialised/num')
        actions = np.array(os.listdir(PATH))
        model = load_model('num_model')

    elif choice == 3:
        PATH = os.path.join('specialised/alpha')
        actions = np.array(os.listdir(PATH))
        model = load_model('alpha_model')

    elif choice == 4:
        PATH = os.path.join('specialised/gesture')
        actions = np.array(os.listdir(PATH))
        model = load_model('gesture_model')

    elif choice == 0:
        print('Exited!')

    else:
        print('Select a correct option!')

    if choice == 1 or choice == 2 or choice == 3 or choice == 4:
        sentence, keypoints = [' '], []

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access camera.")
            exit()

        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            while cap.isOpened():
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
                
                if keyboard.is_pressed(' '):
                    sentence = [' ']

                # if keyboard.is_pressed('c'):
                #     context = ' '.join(sentence)
                #     text = "paraphrase: "+context + " </s>"
                #     encoding = tokenizer.encode_plus(text,max_length =10, padding=True, return_tensors="pt")
                #     input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
                #     attmodel.eval()
                #     diverse_beam_outputs = attmodel.generate(
                #         input_ids=input_ids,attention_mask=attention_mask,
                #         early_stopping=True,
                #         num_beams=5,
                #         num_beam_groups = 5,
                #         num_return_sequences=5,
                #         diversity_penalty = 0.70
                #     )
                #     sent = tokenizer.decode(diverse_beam_outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                #     sentence = sent[18:].lower()

                textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (image.shape[1] - textsize[0]) // 2
                    
                cv2.putText(image, ' '.join(sentence), (text_X_coord, 470), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Camera', image)
                cv2.waitKey(1)
                if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1 or keyboard.is_pressed('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()