import gradio as gr
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import numpy as np

# Load the pre-trained model and preprocessor (feature extractor)
model_tuberculosis = ViTForImageClassification.from_pretrained("./jitsa_tuberculosis_model")
feature_extractor = ViTFeatureExtractor.from_pretrained("./jitsa_feature_extractor")
def classify_image(image):
    # Convert the PIL Image to a format compatible with the feature extractor
    image = np.array(image)
    # Preprocess the image and prepare it for the model
    inputs_tuberculosis = feature_extractor(images=image, return_tensors="pt")
    # Make prediction
    with torch.no_grad():
        outputs_tuberculosis = model_tuberculosis(**inputs_tuberculosis)
        logits_tuberculosis = outputs_tuberculosis.logits
    # Retrieve the highest probability class label index
    predicted_class_idx_tuberculosis = logits_tuberculosis.argmax(-1).item()
    # Define a manual mapping of label indices to human-readable labels
    index_to_label_tuberculosis = {0: "Tuberculosis = NO",1: "Tuberculosis = YES"}
    # Convert the index to the model's class label
    label_tuberculosis = index_to_label_tuberculosis.get(predicted_class_idx_tuberculosis, "Unknown Label")

    return label_tuberculosis


# Create title, description and article strings
title = "Automated Classification of Tuberculosis using Machine Learning"
description = "Upload your lungs Radiograph to find out if you are having Tuberculosis"

css_code = ".gradio-container {background: url(https://media.istockphoto.com/vectors/lungs-low-poly-blue-vector-id1039566852?k=6&m=1039566852&s=170667a&w=0&h=NBNf36zqI9cpSqpM0sw-PDq-J6mm55vciEKY9-43wWA=); background-size: cover;}"

# Create Gradio interface
iface = gr.Interface(fn=classify_image, 
                     inputs=gr.Image(),  # Accepts image of any size
                     outputs=gr.Label(),
                     title=title,
                     description=description,
                     css=css_code
                    )

# Launch the app 
iface.launch()

css_code = f"""
.gradio-container {{
  background-image: url('{background_image_path}');
  background-size: cover;  /* Ensure image covers the container */
  background-position: center;  /* Center the image */
  /* Add other styling options (e.g., padding, color) */
}}
"""
