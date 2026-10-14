
# ============================================================
# BRAIN TUMOR MRI CLASSIFICATION - STREAMLIT
# ============================================================

import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

# ------------------------------------------------------------
# 1. Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain MRI Tumor Classification")
st.write(
    "Upload a brain MRI image to see the prediction "
    "from the trained EfficientNetB0 model."
)

st.caption(
    "For educational and research use only. "
    "This application does not provide a medical diagnosis."
)

# ------------------------------------------------------------
# 2. Model file paths
# ------------------------------------------------------------

DATA_DIR = r"C:\Users\Rahila\Downloads\brain_tumor_dataset"

MODEL_PATH = os.path.join(DATA_DIR, "efficientnet_model.keras")
CLASS_NAMES_PATH = os.path.join(DATA_DIR, "class_names.json")

IMG_SIZE = (224, 224)

# ------------------------------------------------------------
# 3. Load model and class names
# ------------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as file:
        return json.load(file)

if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found:\n{MODEL_PATH}")
    st.info("Run the Jupyter training notebook first.")
    st.stop()

if not os.path.isfile(CLASS_NAMES_PATH):
    st.error(f"Class names file not found:\n{CLASS_NAMES_PATH}")
    st.info("Run the Jupyter training notebook first.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    CLASS_NAMES = load_classes(CLASS_NAMES_PATH)
except Exception as error:
    st.error(f"Failed to load model or class names: {error}")
    st.stop()

if len(CLASS_NAMES) != model.output_shape[-1]:
    st.error(
        "The number of class names does not match the model output. "
        "Make sure the model and class_names.json come from the "
        "same training run."
    )
    st.stop()

st.success("Model loaded successfully.")

# ------------------------------------------------------------
# 4. Upload an MRI image
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload brain MRI image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded MRI image")
        st.image(image, use_container_width=True)

        if st.button("Analyze MRI", type="primary"):
            with st.spinner("Analyzing image..."):

                # Resize to the model's expected dimensions.
                resized_image = image.resize(
                    IMG_SIZE,
                    Image.Resampling.BILINEAR
                )

                # Convert to float32; retain 0-255 pixel values.
                # EfficientNetB0 contains its own rescaling layer.
                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index])

            st.subheader("Prediction")
            st.markdown(f"### {predicted_class}")
            st.metric(
                "Model confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Scores by class")

            score_data = {
                name: float(score)
                for name, score in zip(CLASS_NAMES, predictions)
            }

            st.bar_chart(score_data)

            with st.expander("Show detailed prediction scores"):
                for name, score in score_data.items():
                    st.write(f"**{name}:** {score * 100:.2f}%")

            st.warning(
                "This prediction is not a diagnosis. Model confidence "
                "is not the same as clinical certainty. A qualified "
                "radiologist or clinician must interpret MRI findings."
            )

    except Exception as error:
        st.error(f"Unable to process the image: {error}")

else:
    st.info("Choose an MRI image above to begin.")

st.divider()
st.caption("Brain MRI classifier | EfficientNetB0")
