import numpy as np
from classifiers import *
from pipeline import *
import json
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def loss_matrix(y_true, y_pred):
    return 1.0 - accuracy(y_true, y_pred)

def accuracy(y_true, y_pred):
    return np.mean(np.equal(y_true, np.round(y_pred)))

with open('data/metadata.json') as f:
    ground_truth = json.load(f)
# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')


# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
    'test_images',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary',
    subset='training')

# 3 - Predict
X, y = generator.next()
print('Predicted :', classifier.predict(X), '\nReal class :', y)

# 4 - Prediction for a video dataset

classifier.load('weights/Meso4_F2F.h5')


true_labels = []
pred_labels = []

predictions = compute_accuracy(classifier, 'data')
for video_name in predictions:
    prediction_value = predictions[video_name][0]
    video_name = video_name + '.mp4'
    ground_truth_value = ground_truth[video_name]['label']

    true_labels.append(1.0 if ground_truth_value == "FAKE" else 0.0)
    pred_labels.append(prediction_value)

y_true = np.array(true_labels)
y_pred = np.array(pred_labels)

loss = K.eval(K.mean(K.binary_crossentropy(y_true, y_pred)))
acc = accuracy(y_true, y_pred)
print("Loss: ", loss)
print("Accuracy: ", acc)

conf_matrix = confusion_matrix(y_true, np.round(y_pred))
print("Confusion Matrix: \n", conf_matrix)


# predictions = compute_accuracy(classifier, 'data')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(
#         video_name), predictions[video_name][0])

# correct_predictions = 0
# total_predictions = len(predictions)

# for video_name, prediction in predictions.items():
#     prediction_value = prediction[0]
#     video_name = video_name + '.mp4'
#     ground_truth_value = ground_truth[video_name]['label']

#     if prediction_value == 1.0 and ground_truth_value == "FAKE":
#         correct_predictions += 1

# accuracy = correct_predictions / total_predictions
# print('FINAL ACCURACY: ', accuracy)
