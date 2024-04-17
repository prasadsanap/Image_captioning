import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from transformers import pipeline

# Load the image captioning model
def load_image_captioning_model(model_path):
    model = tf.keras.models.load_model(r"C:\Users\Prasad\Desktop\app\Image_captioning\best_model.h5")
    return model

# Load the text simplification pipeline
simplifier = pipeline(model='haining/sas_baseline')

# Function to preprocess and caption the uploaded image
def caption_image(model, image):
    # Load and preprocess the image
    image = Image.open(image).convert("RGB")
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.asarray(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    
    # Generate caption using the model
    caption = model.predict(image)
    return caption

def main():
    st.title("Image Captioning and Text Simplification App")

    # Load the image captioning model
    model_path = "path/to/your/best_model.h5"  # Replace with the path to your trained model
    image_captioning_model = load_image_captioning_model(model_path)

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Home", "Image Captioning", "Text Simplification"])

    if page == "Home":
        st.write("Welcome to the Image Captioning and Text Simplification App!")

    elif page == "Image Captioning":
        st.header("Image Captioning")
        # File uploader for image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Process the image and generate a caption
            caption = caption_image(image_captioning_model, uploaded_image)
            st.image(uploaded_image, caption=caption, use_column_width=True)

    elif page == "Text Simplification":
        st.header("Text Simplification")
        # Text input for text simplification
        input_text = st.text_area("Enter text")
        if st.button("Simplify"):
            # Process the text using the simplification pipeline
            simplified_text = simplifier(input_text)
            st.write("Simplified text:", simplified_text)

if __name__ == "__main__":
    main()
