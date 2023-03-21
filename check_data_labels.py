import json
import os

# Path to the directory containing the video files
video_dir = 'data/'

# Path to the JSON file
json_file = 'metadata.json'

# Read the JSON file and create a dictionary with file names and labels
with open(json_file, 'r') as f:
    data = json.load(f)

labels = {}
for filename, label_data in data.items():
    if 'label' in label_data:
        label = label_data['label']
        if label == 'REAL' or label == 'FAKE':
            labels[filename] = label
        else:
            print(f'Error: Invalid label {label} for {filename}')
    else:
        print(f'Error: No label found for {filename}')
        # Remove the file if it does not have a label
        os.remove(os.path.join(video_dir, filename))

# Check each video file in the directory
for filename in os.listdir(video_dir):
    if filename.endswith('.mp4'):
        # Check if the file name is in the dictionary of labels
        if filename in labels:
            label = labels[filename]
            print(f'{filename}: {label}')
        else:
            # Remove the file if it does not have a label
            os.remove(os.path.join(video_dir, filename))
