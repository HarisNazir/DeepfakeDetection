import os
import csv

# Set the directory path
directory = 'Face_only_data_split/Test'

# Set the starting and ending numbers for renaming
start_number = 475
end_number = 645

# Open the data.csv file for updating
with open('data.csv', 'r') as file:
    csv_reader = csv.reader(file)
    csv_data = [row for row in csv_reader]

# Loop through all the files in the directory
for count, filename in enumerate(os.listdir(directory)):
    # Create the new filename with the specified number range
    new_filename = f"{start_number + count}.mp4"
    
    # Join the directory path with the old filename
    old_file_path = os.path.join(directory, filename)
    
    # Join the directory path with the new filename
    new_file_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)
    
    # Update the filename in the csv data
    for row in csv_data:
        if filename in row[0]:
            row[0] = new_filename
    
# Write the updated csv data back to the file
with open('data.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(csv_data)