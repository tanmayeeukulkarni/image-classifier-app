import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

st.title("Image Classification using MobileNetV2")

model = MobileNetV2(weights="imagenet")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    results = decode_predictions(preds, top=3)[0]

    st.subheader("Predictions:")
    for r in results:
        st.write(f"{r[1]} : {round(r[2]*100,2)}%")
