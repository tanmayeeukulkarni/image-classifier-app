
import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

st.title("Image Classification using Deep Learning (Lightweight Model)")

# Load ONNX model
session = ort.InferenceSession("model.onnx")

def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image).astype("float32") / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    input_data = preprocess(image)

    outputs = session.run(None, {"input": input_data})
    pred = np.argmax(outputs[0])

    st.subheader(f"Predicted Class Index: {pred}")
