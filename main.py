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
from keras.applications.resnet50 import ResNet50

# Load the metadata
with open('data/metadata.json', 'r') as f:
    metadata = json.load(f)

# Split the data into train and test sets
train_files = [k for k, v in metadata.items() if v['split'] == 'train']
test_files = [k for k, v in metadata.items() if v['split'] == 'test']

# Preprocess the data


def preprocess(video_file):
    # Split the video into frames
    frames = split_into_frames(video_file)
    
    # Detect faces in the frames
    faces = detect_faces(frames)
    
    # Remove frames without faces
    frames = [f for f in frames if f is not None]
    
    # Crop the first 300 frames from each video
    frames = frames[:300]
    
    # Convert frames to numpy array
    X = np.array(frames)
    
    # Use ResNeXt-50 CNN to extract features
    X = extract_features(X)
    
    return X

def split_into_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_faces(frames):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces.append(face)
    return faces


def extract_features(preprocessed_data):
    # Create an instance of the ResNeXt50 model
    resnext50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    
    # Extract features from the preprocessed data using the ResNeXt50 model
    features = resnext50_model.predict(preprocessed_data)
    
    # Flatten the extracted features
    features = features.reshape(features.shape[0], -1)
    
    return features

def create_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(300, num_features)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model