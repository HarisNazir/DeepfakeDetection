import cv2
import dlib
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def align_face(frame, landmarks):
    # Get the landmarks of the eyes and mouth
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]

    # Get the center of the eyes
    eyes_center = ((left_eye[0].x + right_eye[3].x) // 2, (left_eye[0].y + right_eye[3].y) // 2)

    # Get the angle between the eyes
    dy = right_eye[1].y - left_eye[1].y
    dx = right_eye[1].x - left_eye[1].x
    angle = math.atan2(dy, dx) * 180.0 / math.pi

    # Get the size of the eyes
    eyes_size = int(math.sqrt((dy ** 2) + (dx ** 2)))

    # Get the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Get the aligned face
    aligned_face = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))

    return aligned_face

def train_model(features,labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Create a support vector classifier
    model = SVC(kernel='linear')

    # Train the classifier on the training data
    model.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return model

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load dlib's landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load pre-trained FaceNet model
model = keras.models.load_model('facenet_keras.h5')

# Open video file
video = cv2.VideoCapture('video.mp4')

# Create labels for the data. 1 for real, 0 for deepfake.
labels = np.array([1, 1, 1, 0, 0, 0])

while True:
    # Read frame from video
    ret, frame = video.read()

    # Exit loop if video ends
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Iterate over each face
    for face in faces:
        # Get the landmarks of the face
        landmarks = predictor(gray, face)

        # Align the face using the landmarks
        aligned_face = align_face(frame, landmarks)

        # Extract facial features from aligned face
        features = model.predict(aligned_face[None, ...])

classifier = train_model(features, labels)

# Release video file
video.release()