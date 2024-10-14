# CS5330_lab2

## To prepare the dataset, you should run:
### python3 data_preparation.py
### This script will split the dataset into training and testing sets and save them as texture_dataset.npz

## Next we need to extract GLCM and LBP features, you should run:
### python3 feature_extraction.py
### This script will load data from texture_dataset.npz and extract texture features. The extracted features will be saved as processed_features.npz

## To train the SVM classifier and view the results, you should run:
### python3 train_model.py
### After running, you will see the classification accuracy and a detailed classification report based on both GLCM and LBP features.

## To launch the Gradio interface for real-time texture classificatio, you should run:
### python3 app.py
### This will start a Gradio web interface