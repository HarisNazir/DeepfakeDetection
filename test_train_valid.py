import os
import random
import shutil
import json

# Define the paths
data_dir = 'data'
split_dir = 'data_split'
train_dir = os.path.join(split_dir, 'Train')
test_dir = os.path.join(split_dir, 'Test')
val_dir = os.path.join(split_dir, 'Validation')

# Define the proportions
train_prop = 0.7
test_prop = 0.2
val_prop = 0.1

# Read the labels
labels_file = 'metadata.json'
with open(labels_file) as f:
    labels = json.load(f)

# Create the split directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Initialize counters
num_real = 0
num_fake = 0

# Randomly assign each video to a folder
for filename, label_data in labels.items():
    label = label_data['label']
    rand_num = random.random()
    if label == 'REAL':
        num_real += 1
        if rand_num < train_prop:
            shutil.copy2(os.path.join(data_dir, filename), os.path.join(train_dir, filename))
        elif rand_num < train_prop + test_prop:
            shutil.copy2(os.path.join(data_dir, filename), os.path.join(test_dir, filename))
        else:
            shutil.copy2(os.path.join(data_dir, filename), os.path.join(val_dir, filename))
    elif label == 'FAKE':
        num_fake += 1
        if rand_num < train_prop:
            shutil.copy2(os.path.join(data_dir, filename), os.path.join(train_dir, filename))
        elif rand_num < train_prop + test_prop:
            shutil.copy2(os.path.join(data_dir, filename), os.path.join(test_dir, filename))
        else:
            shutil.copy2(os.path.join(data_dir, filename), os.path.join(val_dir, filename))

# Print some statistics
num_train = len(os.listdir(train_dir))
num_test = len(os.listdir(test_dir))
num_val = len(os.listdir(val_dir))
total = num_train + num_test + num_val
print(f'Total videos: {total}')
print(f'Number of real videos: {num_real}')
print(f'Number of fake videos: {num_fake}')
print(f'Train set size: {num_train} ({num_train/total:.2%})')
print(f'Test set size: {num_test} ({num_test/total:.2%})')
print(f'Test set size: {num_val} ({num_val/total:.2%})')

