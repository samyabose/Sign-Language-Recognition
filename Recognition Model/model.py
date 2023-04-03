import os
import numpy as np
import seaborn as sns
from sklearn import metrics
from itertools import product
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
sequences = 30
frames = 10
label_map = {label:num for num, label in enumerate(actions)}
landmarks, labels = [], []

for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

X, Y = np.array(landmarks), to_categorical(labels).astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

model = Sequential()
model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(10,126)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, epochs=200)

model.save('model')

predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(Y_test, axis=1)

accuracy = metrics.accuracy_score(test_labels, predictions)
print('Accuracy:\t', accuracy)

yhat = model.predict(X_test)
ytrue = np.argmax(Y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

reversed_labels = {label_map[k]:k for k in label_map}
ytrue = [reversed_labels[elem] for elem in ytrue]
yhat = [reversed_labels[elem] for elem in yhat]

mat = confusion_matrix(ytrue, yhat)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(mat.T, square = True, annot = True, fmt = "d", xticklabels = np.unique(ytrue), yticklabels = np.unique(ytrue))
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.savefig("output/confusion_matrix.png")

print(metrics.classification_report(ytrue, yhat, zero_division = 0))