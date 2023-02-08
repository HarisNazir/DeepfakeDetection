import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the metadata
with open('data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Split the data into train and test sets
train_files = [k for k, v in metadata.items() if v['split'] == 'train']
test_files = [k for k, v in metadata.items() if v['split'] == 'test']

# Preprocess the data


def preprocess(video_file):
    # TODO: Preprocess the video file and return the features
    # For example, you can extract the frames from the video and convert them into a numpy array
    return features


X_train = np.array([preprocess(os.path.join('data', f)) for f in train_files])
y_train = np.array([metadata[f]['label'] for f in train_files])
X_test = np.array([preprocess(os.path.join('data', f)) for f in test_files])
y_test = np.array([metadata[f]['label'] for f in test_files])

# One hot encoding of labels
labels = np.unique(y_train)
one_hot_map = {l: i for i, l in enumerate(labels)}
y_train = np.array([one_hot_map[y] for y in y_train])
y_test = np.array([one_hot_map[y] for y in y_test])

# Build the model
model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(len(labels), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32,
                    epochs=10, validation_data=(X_test, y_test))

# Plot the accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot the confusion matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cm, index=[i for i in ["REAL", "FAKE"]], columns=[i for i in ["REAL", "FAKE"]])
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix")
sn.heatmap(df_cm, annot=True)
plt.show()
