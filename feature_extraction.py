# CS5330 lab2
# Jiaqi Liu / Pingqi An
# feature_extraction.py
# 10/13/2024

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def compute_glcm_features(image):
    # Calculate GLCM with multiple distances and angles
    glcm = graycomatrix(
        image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=256, symmetric=True, normed=True
    )
    # Extract different texture properties from the GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]  # Measure of local variations
    correlation = graycoprops(glcm, 'correlation')[0, 0]  # Measure of how correlated a pixel is to its neighbors
    energy = graycoprops(glcm, 'energy')[0, 0]  # Measure of uniformity in the GLCM
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]  # Measure of similarity of pixel pairs
    return [contrast, correlation, energy, homogeneity]

def compute_lbp_features(image, radii=[2, 3, 4], points=[8, 16, 24]):
    # Initialize an empty list to store feature vectors
    features = []
    # Iterate over each radius and number of points
    for radius in radii:
        for n_points in points:
            # Calculate LBP for each combination
            lbp = local_binary_pattern(
                image, n_points, radius, method='uniform'
            )
            hist, _ = np.histogram(
                lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
            )
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalize histogram
            features.extend(hist)  # Add this histogram to the feature list
    return features

def load_data(file_path):
    # Load data from a specified file
    dataset = np.load(file_path)
    return (dataset['X_train'], dataset['X_test'], 
            dataset['y_train'], dataset['y_test'])

def save_features(file_path, glcm_train, glcm_test, lbp_train, lbp_test, y_train, y_test):
    # Save extracted features for further use
    np.savez(
        file_path,
        X_train_glcm=glcm_train,
        X_test_glcm=glcm_test,
        X_train_lbp=lbp_train,
        X_test_lbp=lbp_test,
        y_train=y_train,
        y_test=y_test
    )

# Load images and labels
train_images, test_images, train_labels, test_labels = load_data('texture_dataset.npz')

# Extract GLCM and LBP features
train_features_glcm = [compute_glcm_features(img) for img in train_images]
test_features_glcm = [compute_glcm_features(img) for img in test_images]
train_features_lbp = [compute_lbp_features(img) for img in train_images]
test_features_lbp = [compute_lbp_features(img) for img in test_images]

# Save the processed features
save_features(
    'processed_features.npz', 
    train_features_glcm, test_features_glcm, 
    train_features_lbp, test_features_lbp, 
    train_labels, test_labels
)
