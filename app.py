import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="PneumoVision AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------

MODEL_PATH = "mobilenetv2_original.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------------------------
# HEADER
# ---------------------------------

st.title("🩺 PneumoVision AI")
st.subheader("AI-Powered Clinical Decision Support System")

st.write(
    """
    This system analyzes chest X-ray images using a MobileNetV2-based
    deep learning model and provides automated pneumonia screening
    predictions with confidence scores.
    """
)

st.markdown("---")

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.header("Model Information")

st.sidebar.success("Model: MobileNetV2")

st.sidebar.info("""
Input Size: 160 × 160

Classes:
• Normal
• Pneumonia

Framework:
• TensorFlow / Keras
""")

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

st.info(
    "Upload a chest X-ray image (JPG, JPEG, or PNG) to generate a pneumonia screening prediction."
)

# ---------------------------------
# PREDICTION PIPELINE
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            width=450
        )

        width, height = image.size

        st.markdown("### Image Information")

        c1, c2 = st.columns(2)

        c1.metric("Width", width)
        c2.metric("Height", height)

    # ---------------------------------
    # PREPROCESS
    # ---------------------------------

    img = np.array(image)

    img = cv2.resize(
        img,
        (160, 160)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # ---------------------------------
    # PREDICTION
    # ---------------------------------

    prediction = model.predict(img)

    score = float(prediction[0][0])

    with col2:

        st.subheader("Prediction Result")

        if score > 0.5:

            confidence = score * 100
            result = "PNEUMONIA"

            st.error(
                "Prediction: PNEUMONIA"
            )

        else:

            confidence = (1 - score) * 100
            result = "NORMAL"

            st.success(
                "Prediction: NORMAL"
            )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        # ---------------------------------
        # DASHBOARD METRICS
        # ---------------------------------

        st.markdown("### Model Summary")

        m1, m2, m3 = st.columns(3)

        m1.metric("Model", "MobileNetV2")
        m2.metric("Classes", "2")
        m3.metric("Input", "160×160")

        # ---------------------------------
        # CONFIDENCE LEVEL
        # ---------------------------------

        if confidence >= 90:
            st.success("High Confidence Prediction")

        elif confidence >= 75:
            st.warning("Moderate Confidence Prediction")

        else:
            st.error("Low Confidence Prediction")

    # ---------------------------------
    # CLINICAL INTERPRETATION
    # ---------------------------------

    st.markdown("---")

    st.subheader("Clinical Interpretation")

    if result == "PNEUMONIA":

        st.warning(
            """
            The model detected radiographic patterns commonly associated
            with pneumonia. Further clinical evaluation and expert review
            are recommended.
            """
        )

    else:

        st.success(
            """
            No strong radiographic evidence of pneumonia was identified
            by the model. Clinical correlation is recommended.
            """
        )

    # ---------------------------------
    # RECOMMENDATION
    # ---------------------------------

    st.subheader("Clinical Recommendation")

    if result == "PNEUMONIA":

        st.error(
            """
            Recommend further diagnostic assessment by a qualified
            healthcare professional.
            """
        )

    else:

        st.info(
            """
            No significant signs of pneumonia detected by the model.
            """
        )

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")

st.caption(
    """
    Disclaimer:
    This Clinical Decision Support System is intended for educational
    and research purposes only. Predictions should not replace
    professional medical diagnosis.
    """
)
