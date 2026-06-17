import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Pneumonia CDSS",
    page_icon="🩺",
    layout="wide"
)

MODEL_PATH = "mobilenetv2_original.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🩺 PneumoVision AI")
st.subheader("AI-Powered Clinical Decision Support System")
st.write(
    "This system analyzes chest X-ray images using a MobileNetV2-based deep learning model and provides pneumonia screening predictions with confidence scores."
)

st.markdown("---")

st.sidebar.header("Model Information")

st.sidebar.success("Model : MobileNetV2")

st.sidebar.info("""
Input Size: 160 × 160

Classes:
• Normal
• Pneumonia

Framework:
• TensorFlow / Keras
""")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

st.info(
    "Upload a chest X-ray image (JPG, JPEG, or PNG) to generate a pneumonia screening prediction."
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    img = np.array(image)

    img = cv2.resize(
        img,
        (160,160)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(img)

    score = float(prediction[0][0])

    st.markdown("---")

    st.subheader("Prediction Result")

    if score > 0.5:

        confidence = score * 100

        st.error("Prediction : PNEUMONIA")

    else:

        confidence = (1-score) * 100

        st.success("Prediction : NORMAL")

    st.metric(
        "Confidence Score",
        f"{confidence:.2f}%"
    )
    st.markdown("---")

    st.subheader("Clinical Recommendation")

    if score > 0.5:
        st.warning(
        "Potential signs of pneumonia detected. Further clinical evaluation is recommended."
    )
    else:
        st.info(
        "No significant signs of pneumonia detected by the model."
    )