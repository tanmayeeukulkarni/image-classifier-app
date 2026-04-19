
import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("Image Classification using Deep Learning")

# Load model (lightweight, auto-download)
classifier = pipeline("image-classification")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    result = classifier(image)

    st.subheader("Predictions:")
    for r in result[:3]:
        st.write(f"{r['label']} : {round(r['score']*100, 2)}%")
