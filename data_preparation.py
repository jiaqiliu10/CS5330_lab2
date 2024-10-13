# CS5330 lab2
# Jiaqi Liu / Pingqi An
# data_preparation.py
# 10/13/2024

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Specify relative paths to the grass and wood image folders
grass_folder = './grass'
wood_folder = './wood'
resize_dims = (200, 200)  # Set a uniform image size

def fetch_images(directory, label_value):
    image_data = []  # List to store image data
    image_labels = []  # List to store image labels
    file_list = os.listdir(directory)  # Get all files in the directory
    index = 0
    # Use a while loop to iterate through the files
    while index < len(file_list):
        file = file_list[index]
        # Read the image and convert it to grayscale
        image = cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE)
        if image is not None:# Check if the image was successfully loaded
            resized_image = cv2.resize(image, resize_dims)  # Resize the image
            image_data.append(resized_image)  # Add the resized image to the list
            image_labels.append(label_value)  # Add the corresponding label to the list
        index += 1
    return image_data, image_labels

# Load grass and wood image data
grass_data, grass_tags = fetch_images(grass_folder, 0)  # 0 represents the label for grass
wood_data, wood_tags = fetch_images(wood_folder, 1)    # 1 represents the label for wood

# Combine grass and wood image data and their labels into one dataset
all_images = np.array(grass_data + wood_data)
all_labels = np.array(grass_tags + wood_tags)
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=42
) # test_size=0.3 means 30% of the data is used for testing, 70% for training
# random_state=42 ensures the split is the same every time for reproducibility

# Save the split data into a file for later use
np.savez(
    'texture_dataset.npz',
    X_train=train_images,
    X_test=test_images,
    y_train=train_labels,
    y_test=test_labels
)
# Print a message indicating data preparation is complete and the size of each set
print(
    "Data preparation is complete.",
    "Training set size:", len(train_images),
    "Testing set size:", len(test_images)
)