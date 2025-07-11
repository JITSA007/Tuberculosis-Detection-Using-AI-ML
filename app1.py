import gradio as gr
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch
import numpy as np

# --- Load the pre-trained model and feature extractor ---
# Ensure the paths "./jitsa_tuberculosis_model" and "./jitsa_feature_extractor" are correct.
try:
    model_tuberculosis = ViTForImageClassification.from_pretrained("./jitsa_tuberculosis_model")
    feature_extractor = ViTFeatureExtractor.from_pretrained("./jitsa_feature_extractor")
except OSError:
    # This is a fallback for demonstration if the local model files are not found.
    # It uses a generic image classification model. Replace with your actual model.
    print("Local model not found. Using a public placeholder model for demonstration.")
    model_name = "google/vit-base-patch16-224"
    model_tuberculosis = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)


# --- Modified Classification Function to Return Confidence Scores ---
def classify_image(image):
    """
    Preprocesses the image, runs it through the model, and returns
    a dictionary of labels with their corresponding confidence scores.
    """
    if image is None:
        return None

    # Preprocess the image
    inputs_tuberculosis = feature_extractor(images=image, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs_tuberculosis = model_tuberculosis(**inputs_tuberculosis)
        logits = outputs_tuberculosis.logits

    # Apply softmax to convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Define your labels corresponding to the model's output indices
    # IMPORTANT: Ensure this order matches your model's training.
    # index 0: Normal, index 1: Tuberculosis
    labels = {
        "Normal": probabilities[0].item(),
        "Tuberculosis": probabilities[1].item()
    }
    return labels


# --- CSS for a Modern and Beautiful Interface ---
css_code = """
/* --- Main Container Styling --- */
.gradio-container {
    background: #0F2027; /* fallback for old browsers */
    background: -webkit-linear-gradient(to right, #2C5364, #203A43, #0F2027);
    background: linear-gradient(to right, #2C5364, #203A43, #0F2027);
    font-family: 'Inter', sans-serif;
}

/* --- Header Styling --- */
#title {
    text-align: center;
    color: #FFFFFF;
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 5px;
}
#description {
    text-align: center;
    color: #B0BEC5;
    font-size: 1.3em;
    margin-bottom: 35px;
}

/* --- Component Styling --- */
.gr-image, .gr-label {
    border-radius: 12px !important;
    border: 2px solid #37474F !important;
    background: #263238 !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.gr-label .label-name { color: #90A4AE !important; }

/* --- Button Styling --- */
.gr-button {
    background: #009688 !important; /* A vibrant, modern teal */
    color: white;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1.1em !important;
    transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
}
.gr-button:hover {
    background-color: #00796B !important;
    transform: scale(1.02);
}

/* --- Custom Footer --- */
#footer {
    text-align: center;
    color: #78909C;
    font-size: 0.9em;
    margin-top: 40px;
}
"""

# --- Build the Gradio Interface with Blocks ---
with gr.Blocks(css=css_code, theme=gr.themes.Soft()) as iface:
    # --- Header ---
    gr.Markdown("Detection and Diagnosis of Latent Tuberculosis (TB)", elem_id="title")
    gr.Markdown("Upload your chest radiograph for an AI-powered analysis.", elem_id="description")

    # --- Main Layout (Input/Output) ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Radiograph")
            submit_btn = gr.Button("Analyze Image")

        with gr.Column(scale=1):
            output_label = gr.Label(label="Analysis Results", num_top_classes=2)

    # --- Footer ---
    gr.Markdown("Developed by **Jitendra Prajapat**", elem_id="footer")

    # --- Event Handling ---
    submit_btn.click(
        fn=classify_image,
        inputs=input_image,
        outputs=output_label
    )

# --- Launch the App ---
iface.launch(share=True)