import numpy as np
from classifiers import *
from pipeline import *
import json
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import matplotlib as plt

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


predictions = compute_accuracy(classifier, 'data')

# Get the ground truth labels
ground_truth_labels = []
for video_name, prediction in predictions.items():
    video_name = video_name + '.mp4'
    ground_truth_value = ground_truth[video_name]['label']
    ground_truth_labels.append(1.0 if ground_truth_value == "FAKE" else 0.0)

# Get the predicted labels
predicted_labels = [prediction[0] for prediction in predictions.values()]

# Calculate the confusion matrix
conf_mat = confusion_matrix(ground_truth_labels, predicted_labels)

# Visualize the confusion matrix
plt.imshow(conf_mat, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.xticks([0, 1], ['Real', 'Fake'])
plt.yticks([0, 1], ['Real', 'Fake'])
plt.colorbar()
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        plt.text(j, i, conf_mat[i, j], ha='center', va='center', color='black')
plt.show()

# Calculate the loss rate
loss_rate = 1.0 - accuracy

# Plot the accuracy and loss rate
plt.plot([accuracy, loss_rate])
plt.title('Accuracy and Loss Rate')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.xticks([0, 1], ['Accuracy', 'Loss Rate'])
plt.show()


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
