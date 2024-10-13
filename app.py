# CS5330 lab2
# Jiaqi Liu / Pingqi An
# gradio_app.py
# 10/13/2024

import gradio as gr
import numpy as np
import cv2
from feature_extraction import compute_glcm_features, compute_lbp_features  
from train_model import glcm_svm_model, lbp_svm_model


def classify_image(image, method):
    # Define image size for resizing
    img_size = (200, 200)
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to (200, 200)
    image = cv2.resize(image, img_size)
    
    # Check if the user has selected a method
    if method not in ["GLCM", "LBP"]:
        return "Please select a method (GLCM or LBP)."
    
    if method == "GLCM":
        # Extract features using GLCM and classify
        features = compute_glcm_features(image)
        label = glcm_svm_model.predict([features])[0]
    else:
        # Extract features using LBP and classify
        features = compute_lbp_features(image)
        label = lbp_svm_model.predict([features])[0]
    
    return "Grass" if label == 0 else "Wood"

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=[gr.Image(), gr.Radio(["GLCM", "LBP"])],
    outputs="text",
    title="Texture Classification of Grass and Wood",
    description="Upload an image and select a method to classify it as Grass or Wood."
)

# Launch the interface
iface.launch(share=True)


