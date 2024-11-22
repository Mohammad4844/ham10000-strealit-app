import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os

# Define the class names (HAM10000 classes)
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

class_map = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

def load_model():
    model = torch.load("model.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Transform for the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def main():
    # Check if the user has acknowledged the warning
    if "acknowledged" not in st.session_state:
        st.session_state.acknowledged = False

    # If not acknowledged, display the warning
    if not st.session_state.acknowledged:
        st.title("Content Display Warning")
        st.write("""
        This app contains medical content and images that may be sensitive or distressing for some users. 
        Please proceed only if you are comfortable viewing this content.
        """)

        if st.button("I Acknowledge and Wish to Proceed"):
            st.session_state.acknowledged = True
            
    else:
        # Main app content
        st.title("Skin Cancer Classifier")

        # Preloaded image options
        image_options = {
            "Image 1": "images/img1.jpg",
            "Image 2": "images/img2.jpg",
            "Image 3": "images/img3.jpg",
            "Image 4": "images/img4.jpg",
        }

        # Dropdown to select an image
        selected_image_label = st.selectbox("Select an image:", list(image_options.keys()))

        # Load the selected image
        selected_image_path = image_options[selected_image_label]
        image = Image.open(selected_image_path)

        # Display the selected image
        st.image(image, caption=f"Selected: {selected_image_label}")

        # Predict when the button is clicked
        if st.button("Get Prediction"):
            # Preprocess the image
            input_tensor = preprocess_image(image)

            # Pass through the model
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_class = torch.max(output, 1)

            # Display the prediction
            predicted_label = class_names[predicted_class.item()]
            st.subheader(f"Predicted Class: {predicted_label} ({class_map[predicted_label]})")

if __name__ == "__main__":
    main()