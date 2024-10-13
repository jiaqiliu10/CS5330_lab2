# CS5330 lab2
# Jiaqi Liu / Pingqi An
# train_model.py
# 10/13/2024

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load extracted features
feature_data = np.load('processed_features.npz')
train_features_glcm = feature_data['X_train_glcm']
test_features_glcm = feature_data['X_test_glcm']
train_features_lbp = feature_data['X_train_lbp']
test_features_lbp = feature_data['X_test_lbp']
train_labels = feature_data['y_train']
test_labels = feature_data['y_test']

# Train SVM classifier using GLCM features
glcm_svm_model = SVC(kernel='linear')
glcm_svm_model.fit(train_features_glcm, train_labels)
predictions_glcm = glcm_svm_model.predict(test_features_glcm)

# Train SVM classifier using LBP features
lbp_svm_model = SVC(kernel='linear')
lbp_svm_model.fit(train_features_lbp, train_labels)
predictions_lbp = lbp_svm_model.predict(test_features_lbp)

# Output classification results
print("GLCM-based SVM Accuracy:", accuracy_score(test_labels, predictions_glcm))
print("LBP-based SVM Accuracy:", accuracy_score(test_labels, predictions_lbp))
print("GLCM Classification Report:\n", classification_report(test_labels, predictions_glcm))
print("LBP Classification Report:\n", classification_report(test_labels, predictions_lbp))
