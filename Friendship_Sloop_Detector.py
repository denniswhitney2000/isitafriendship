import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

from pathlib import Path
from dotenv import load_dotenv

# # Suppress only the single warning from urllib3.
# import urllib3
# urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from the .env file (if present)
load_dotenv()

# Define the config class
class CFG:

    # Define the directory to store the images
    DATA_DIR = Path(os.environ['DATA_DIR'])
    
    # Set the number of batchs for processing
    BATCH_SIZE = int(os.environ['BATCH_SIZE'])

    # Epocs for model training
    NUM_EPOCHS = int(os.environ['NUM_EPOCHS'])
    
    MODEL_PATH = Path(os.environ['MODEL_PATH'])


# Load the model in SavedModel format
model = tf.keras.models.load_model(CFG.MODEL_PATH)

# Set up Streamlit app title
st.title("Friendship Sloop Detection App")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Image preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input
    img = np.array(img) / 255.0     # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and make prediction
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)

    # Display result
    if prediction[0][0] > 0.5:
        st.write("### Friendship Sloop detected!")
    else:
        st.write("### No Friendship Sloop detected.")
