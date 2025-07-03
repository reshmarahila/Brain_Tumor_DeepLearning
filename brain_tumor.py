import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Constants
MODEL_PATH = "efficientnet_model.keras"
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = (224, 224)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Preprocess function
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload a brain MRI image and the model will classify it into one of the 4 tumor types.")

uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Classifying..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0]
            pred_index = np.argmax(prediction)
            pred_label = CLASS_NAMES[pred_index]
            confidence = prediction[pred_index]

        st.success(f"🩺 **Prediction:** {pred_label}")
        st.write(f"📊 **Confidence:** {confidence * 100:.2f}%")
        st.bar_chart({CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))})
else:
    st.info("Please upload a brain MRI image to get started.")
