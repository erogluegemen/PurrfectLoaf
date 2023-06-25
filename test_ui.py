import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import keras.models as models
import keras.preprocessing.image as image

st.set_page_config(
  page_title = 'Cat/Loaf Classification',
  page_icon = '🐱',
  )
# Load your trained model
model = models.load_model('3x3x64-catvsloaf.model')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Specify the desired width for displaying images
DISPLAY_WIDTH = 200

# Create a function to make predictions on user-uploaded images
def classify_image(image):
    img = image.resize((100, 100))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_category_index = int(prediction[0][0])
    predicted_category = 'loaf 🥖' if predicted_category_index == 1 else 'cat 🐈'

    return predicted_category

# Resize the image while maintaining its aspect ratio
def resize_image(image, width):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    height = int(width / aspect_ratio)
    resized_image = image.resize((width, height))
    return resized_image

# Create the Streamlit web app
def main():
    st.title("Cat-Loaf Classification")
    st.write("### Upload an image and I'll predict if it's a cat 🐈 or a loaf 🥖!")

    # Create a file uploader in the app
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Make a prediction when a file is uploaded
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        resized_img = resize_image(img, DISPLAY_WIDTH)
        st.image(resized_img, use_column_width=True)
        pred = classify_image(resized_img)
        st.write("## Prediction:", pred)

# Run the app
if __name__ == '__main__':
    main()
