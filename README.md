# DeepfakeDetection
Final Year Project: Deepfake Detection

# Steps to Create a Detection Program
1. Preprocess the dataset by extracting the frames from the videos and aligning the faces in the frames. You can use OpenCV for this step.

2. Extract facial features from the frames using a pre-trained model such as FaceNet or VGGFace. You can use the Tensorflow Keras-OpenCV library to load the pre-trained model and extract the features.

3. Compare the extracted facial features to a database of real images of the person to determine if the video is a deepfake or not.

4. Train a machine learning model using the extracted features and labels as input. You can use a variety of models such as SVM, Random Forest, or CNN.

5. Finally, test the model on unseen data and evaluate its performance.