import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Custom CSS for beautifying the app
st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Center-align the image and text */
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }

    /* Styling for file uploader and button */
    .stFileUploader {
        margin-top: 20px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }

    /* Styling for prediction text */
    .predictions {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
    }
    
    /* Heading styles */
    h1 {
        color: #333;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    h2 {
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title with some styling
st.title("🌟 Beautiful Image Classification App 🌟")
st.write("Upload an image below to classify it using a pre-trained deep learning model!")

# File uploader section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image in a nice layout
    st.subheader("Uploaded Image:")
    image = Image.open(uploaded_file)
    
    # Create a centered container to display the image
    with st.container():
        st.image(image, caption="Your uploaded image", use_column_width=True, clamp=True)

    # Preprocess the image for the model
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions using the model
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the predictions in a more organized and styled manner
    st.subheader("Prediction Results:")
    
    # Create a centered container for predictions
    with st.container():
        for i, pred in enumerate(decoded_predictions):
            st.markdown(
                f"<p class='predictions'>{i+1}: {pred[1]} ({pred[2]*100:.2f}%)</p>",
                unsafe_allow_html=True
            )
else:
    # Display a placeholder message if no file is uploaded yet
    with st.container():
        st.write("Please upload an image to get the classification results.")

# Footer section with some extra space
st.write("")
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.write("Developed with 💖 by [Your Name]")
