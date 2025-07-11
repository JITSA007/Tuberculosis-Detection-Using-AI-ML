
# 🧠 Tuberculosis Detection Using AI/ML

An AI-powered web tool for automated detection of **Tuberculosis (TB)** from chest X-ray images using a fine-tuned **Vision Transformer (ViT)** model. Built with Gradio for ease of use and deployment.

Developed by **Jitendra Prajapat**  
[GitHub](https://github.com/your-username) • [LinkedIn](https://linkedin.com/in/your-profile) • [Email](mailto:your.email@example.com)

---

## 🩺 Project Description

Tuberculosis remains a critical global health issue. This project leverages **transformer-based deep learning** to classify chest radiographs as either:
- ✅ Normal
- 🛑 Tuberculosis

Users can upload a radiograph image through a clean and responsive **Gradio UI**, and receive class probabilities powered by a ViT model.

---

## 🚀 Features

- ✅ Image classification using **ViT (Vision Transformer)**
- 📸 Drag-and-drop chest X-ray upload
- 📊 Real-time prediction with confidence scores
- 🎨 Clean, dark-themed, responsive UI
- 📦 Local model fallback with HuggingFace `from_pretrained()`

---

## 🛠️ Tech Stack

| Component      | Tool/Library        |
|----------------|---------------------|
| Interface      | Gradio              |
| Model          | HuggingFace Transformers |
| Framework      | PyTorch             |
| Image Handling | PIL (Pillow)        |
| Styling        | Custom CSS          |

---

## 🗂️ Project Structure

```
tb-detection-ai-ml/
│
├── app.py                           # Main Gradio app
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── jitsa_tuberculosis_model/      # Fine-tuned ViT model (saved using HuggingFace)
└── jitsa_feature_extractor/       # Feature extractor used for pre-processing
```

---

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/tb-detection-ai-ml.git
cd tb-detection-ai-ml
```

2. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate   # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Ensure the model directories exist:**
```
/jitsa_tuberculosis_model/         # contains config.json, pytorch_model.bin, etc.
/jitsa_feature_extractor/          # contains preprocessor_config.json, etc.
```

---

## 🧪 Running the App

```bash
python app.py
```

Gradio will display:
- A **local URL** for testing on your device.
- A **public shareable URL** if `share=True`.

Upload an X-ray image and receive classification output with confidence scores.

---

## 📷 Screenshots

> *(Add your app screenshots here if available)*

---

## 🧠 Model Details

- Base model: `google/vit-base-patch16-224`
- Fine-tuned on binary chest X-ray dataset (Normal vs. TB)
- Preprocessing handled via HuggingFace `ViTFeatureExtractor`

---

## 📄 License

- **Code**: Licensed under the [MIT License](LICENSE)
- **Model Weights**: For **non-commercial, research & educational use only**. Contact the author for commercial licensing.

---

## 🙋‍♂️ Author

**Jitendra Prajapat**  
Professor | AI/ML Researcher | Developer | Motivational Speaker

---

## ⭐ Acknowledgments

- Hugging Face 🤗 Transformers
- Gradio Team for easy-to-use UI
- Open-source X-ray datasets and medical AI research community

---

## 📬 Feedback or Contributions?

Feel free to [open issues](https://github.com/your-username/tb-detection-ai-ml/issues), contribute, or reach out for collaboration!
