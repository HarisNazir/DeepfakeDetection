import os
import json

# Define the directory path
dir_path = "data_split/Validation"

# Read the labels from the JSON file
with open("metadata.json", "r") as f:
    labels = json.load(f)

# Initialize counters for real and fake videos
num_real = 0
num_fake = 0

# Loop through each file in the directory
for filename in os.listdir(dir_path):
    # Check if the file is a video file
    if filename.endswith(".mp4"):
        # Check the label of the video from the JSON file
        if labels[filename]["label"] == "REAL":
            num_real += 1
        elif labels[filename]["label"] == "FAKE":
            num_fake += 1

# Calculate the percentage of real and fake videos
total = num_real + num_fake
percent_real = (num_real / total) * 100
percent_fake = (num_fake / total) * 100

# Print the results
print("Total videos:", total)
print("Real videos:", num_real, "({:.2f}% of total)".format(percent_real))
print("Fake videos:", num_fake, "({:.2f}% of total)".format(percent_fake))
