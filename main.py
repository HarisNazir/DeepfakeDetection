import os
import cv2
import dlib
import numpy as np

dfdc_path = "data/DeepFake"
videos = os.listdir(dfdc_path)
frames = []

# Detect eyes in the frames
eyes_detected = []

def collect_and_prepare_data():
    #Collect the data

    

    for video in videos:
        vidcap = cv2.VideoCapture(os.path.join(dfdc_path, video))
        success, image = vidcap.read()
        while success:
            frames.append(image)
            success, image = vidcap.read()

    # Preprocess the frames
    for i in range(len(frames)):
        frame = frames[i]
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[i] = frame
        
def detect_eyes():
    # Load the facial landmark detector
    predictor_path = "path/to/shape_predictor_68_face_landmarks.dat" # Should be provided with the dlib library NEED TO RESEARCH
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    
    for frame in frames:
        dets = detector(frame, 1)
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            left_eye = shape.part(36)
            right_eye = shape.part(45)
            eyes_detected.append((left_eye, right_eye))
            
    # OPTIONAL: Draw green dot on eyes (Annotate Eyes)
    for frame, (left_eye, right_eye) in zip(frames, eyes_detected):
        cv2.circle(frame, (left_eye.x, left_eye.y), 2, (0, 255, 0), -1)
        cv2.circle(frame, (right_eye.x, right_eye.y), 2, (0, 255, 0), -1)
        
def measure_blink_rate():
    # Measure blink rate
    blink_rate = []
    
    for i in range(len(frames)):
        left_eye = eyes_detected[i][0]
        right_eye = eyes_detected[i][1]
        left_eye_aspect_ratio = np.linalg.norm(left_eye[1] - left_eye[5]) / np.linalg.norm(left_eye[0] - left_eye[4])
        right_eye_aspect_ratio = np.linalg.norm(right_eye[1] - right_eye[5]) / np.linalg.norm(right_eye[0] - right_eye[4])
        blink_rate.append(left_eye_aspect_ratio + right_eye_aspect_ratio)
        
        
collect_and_prepare_data()
detect_eyes()
measure_blink_rate()