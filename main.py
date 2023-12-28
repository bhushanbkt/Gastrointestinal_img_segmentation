import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the saved model
model = load_model('seg_model')  # Update with the actual path to your trained model file

def preprocess_single_image(image_bytes):
    # Convert BytesIO to numpy array
    img = Image.open(image_bytes).convert('RGB')
    img = np.array(img)

    # Ensure proper color channels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, (256, 256))

    # Normalize pixel values to [0, 1]
    img = img / 255.0

    return img

# Function for making prediction
def make_prediction(model, input_image):
    # Expand dimensions to create batch-like shape (1, height, width, channels)
    input_image = np.expand_dims(input_image, axis=0)

    # Make prediction
    prediction = model.predict(input_image)[0]

    return prediction

# Streamlit app
st.title('Image Segmentation App')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', width=300)

    # Preprocess the image for prediction
    input_image = preprocess_single_image(uploaded_file)

    # Make prediction
    predictions = make_prediction(model, input_image)

    # Convert the predicted mask to binary
    binary_mask = (predictions[:, :, 0] > 0.5).astype(np.uint8)

    # Use 'viridis' colormap for grayscale-like effect
    colored_mask_rgb = plt.get_cmap('viridis')(binary_mask)[:, :, :3]

   # Display the predicted mask with colormap
    fig, ax = plt.subplots(figsize=(1,1))
    ax.imshow(predictions[:,:, 0], cmap='viridis')
    ax.axis('off')
    st.pyplot(fig)

    # Display the captioned image (optional)
    st.text('Segmented Image ')
