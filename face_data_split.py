import os
import shutil

# Set the paths of the input and output directories
face_only_video_dir = './Face_only_data'
output_dir = './Face_only_data_split'

# Create the folders for each section in the output directory
test_folder = os.path.join(output_dir, 'Test')
train_folder = os.path.join(output_dir, 'Train')
validation_folder = os.path.join(output_dir, 'Validation')
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

def determine_section(video_name):
    # Set the path of the data_split directory
    data_split_dir = './data_split'
    
    # Check if the video is in the train folder
    train_dir = os.path.join(data_split_dir, 'train')
    if os.path.isfile(os.path.join(train_dir, video_name)):
        return 'Train'

    # Check if the video is in the test folder
    test_dir = os.path.join(data_split_dir, 'test')
    if os.path.isfile(os.path.join(test_dir, video_name)):
        return 'Test'
    
    # Check if the video is in the validation folder
    validation_dir = os.path.join(data_split_dir, 'validation')
    if os.path.isfile(os.path.join(validation_dir, video_name)):
        return 'Validation'
    
    # If the video is not found in any of the folders, return None
    return None

# Iterate over the videos in the input directory
for video_name in os.listdir(face_only_video_dir):
    # Determine which section the video belongs to based on its name or some other criteria
    section = determine_section(video_name)
    
    # Copy the video to the corresponding section folder in the output directory
    source_path = os.path.join(face_only_video_dir, video_name)
    if section == 'Test':
        destination_path = os.path.join(test_folder, video_name)
    elif section == 'Train':
        destination_path = os.path.join(train_folder, video_name)
    elif section == 'Validation':
        destination_path = os.path.join(validation_folder, video_name)
    else:
        print(f'Error: Video {video_name} does not belong to any section!')
        continue
    shutil.copy(source_path, destination_path)
    
print('Videos split into Test, Train, and Validation sections successfully!')
