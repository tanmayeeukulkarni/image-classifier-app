%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Image Classification using MobileNetV2")

model = tf.keras.applications.MobileNetV2(weights='imagenet')

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    st.subheader("Predictions")
    for label in decoded:
        st.write(label[1], ":", round(label[2]*100, 2), "%")
