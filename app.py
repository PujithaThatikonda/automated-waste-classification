
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Automated Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------------
# PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "mobilenet_best.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMG_SIZE = (160, 160)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------------
# LOAD CLASS NAMES
# -----------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

# -----------------------------------
# APPLICATION TITLE
# -----------------------------------

st.title("♻️ Automated Waste Classification")

st.write(
    "Upload a waste image and let the AI model "
    "identify its category."
)

st.info(
    "Supported categories: Cardboard, Glass, Metal, "
    "Paper, Plastic and Trash."
)

# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# PREDICTION
# -----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Your uploaded waste image",
        width=300
    )

    if st.button("Predict Waste Category"):

        with st.spinner("Analysing image..."):

            # Resize image
            resized_image = image.resize(IMG_SIZE)

            # Convert image to array
            image_array = np.array(resized_image)

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Make prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(predictions[0])

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

        # -----------------------------------
        # DISPLAY RESULTS
        # -----------------------------------

        st.success("Prediction completed!")

        st.subheader("Prediction Result")

        st.markdown(
            f"### ♻️ Predicted Category: {predicted_class.title()}"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        # Display all category probabilities
        st.subheader("Category-wise Confidence")

        for index, category in enumerate(class_names):

            score = float(predictions[0][index]) * 100

            st.write(f"**{category.title()}**")

            st.progress(int(round(score)))

            st.write(f"{score:.2f}%")

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

else:

    st.warning("Please upload an image to get started.")

# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Automated Waste Classification | "
    "Deep Learning using MobileNetV2"
)
